#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 <-> ESP32 USB serial link.

Built so a link problem can never freeze or end a run:
- One background thread owns reading and reconnecting. send_command() never blocks
  on a reconnect; while the link is down commands are dropped and the ESP32's own
  500 ms failsafe stops the motor. When the port comes back, driving resumes.
- The port is opened with DTR/RTS released, so opening or reopening it does not
  reset the ESP32.
- The port is re-detected on every reconnect (prefers /dev/serial/by-id), so a USB
  re-enumeration from ttyUSB0 to ttyUSB1 is followed instead of retried forever.
- A write timeout drops one command; it is not treated as a disconnect.
- Diagnostics: every link event is logged with a timestamp. ESP32 reboots are
  detected (BOOT line or heartbeat uptime going backwards) and the reset reason is
  printed - BROWNOUT means a power problem, not a USB problem.
"""

import glob
import sys
import threading
import time
from typing import Callable, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] 'pyserial' is not installed. Install it with: pip install pyserial", file=sys.stderr)
    raise


USB_SERIAL_KEYWORDS = ("cp210", "ch340", "ch341", "ch910", "ftdi", "usb serial", "usb-serial",
                       "uart", "esp32", "silicon labs", "wch")

_T0 = time.time()


def _log(msg: str):
    print(f"[LINK +{time.time() - _T0:7.2f}s] {msg}", flush=True)


class WROSerialController:
    """USB serial link to the ESP32 motor/steering controller."""

    VALID_COMMANDS = {
        "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP",
        "AUTO_US_ON", "AUTO_US_OFF", "TURN_LEFT", "TURN_RIGHT"
    }

    HEARTBEAT_TIMEOUT = 1.5  # s without any line from the ESP32 -> link considered unhealthy

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.05,
                 auto_connect: bool = True, verbose: bool = False):
        self.requested_port = None if port in (None, "", "AUTO") else port
        self.port: Optional[str] = None
        self.baudrate = baudrate
        self.timeout = timeout
        self.verbose = verbose

        self.serial_conn: Optional[serial.Serial] = None
        self._write_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._need_reopen = False

        self.last_ack: Optional[str] = None
        self.on_ack_callback: Optional[Callable[[str], None]] = None
        self.us_data = {"f": 0, "l": 0, "r": 0, "b": 0}  # kept for older scripts; no sensors on this build

        self.last_rx_time = 0.0
        self._last_uptime_ms = None
        self.stats = {"disconnects": 0, "reconnects": 0, "esp_reboots": 0,
                      "failsafe_stops": 0, "dropped_writes": 0, "last_reset_reason": None,
                      "us_lines": 0}

        if auto_connect:
            self.connect()

    # ------------------------------------------------------------------ port handling
    def find_serial_port(self) -> Optional[str]:
        """Finds the ESP32's USB serial port. Never returns the Pi's own onboard UARTs."""
        if self.requested_port:
            return self.requested_port

        by_id = sorted(glob.glob("/dev/serial/by-id/*"))
        for path in by_id:
            name = path.lower()
            if any(k in name for k in USB_SERIAL_KEYWORDS):
                return path
        if by_id:
            return by_id[0]

        for p in serial.tools.list_ports.comports():
            desc = f"{p.description} {p.manufacturer or ''} {p.hwid}".lower()
            if any(k in desc for k in USB_SERIAL_KEYWORDS) or "ttyusb" in p.device.lower() or "ttyacm" in p.device.lower():
                return p.device
            if sys.platform.startswith("win") and p.device.upper().startswith("COM") and "usb" in desc:
                return p.device

        candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        return candidates[0] if candidates else None

    def _open(self) -> bool:
        target = self.find_serial_port()
        if not target:
            return False
        try:
            ser = serial.Serial()
            ser.port = target
            ser.baudrate = self.baudrate
            ser.timeout = self.timeout
            ser.write_timeout = 0.05
            # Keep DTR/RTS released so opening the port does not pulse the ESP32 reset line.
            ser.dtr = False
            ser.rts = False
            ser.open()
            time.sleep(0.1)
            ser.reset_input_buffer()
        except (serial.SerialException, OSError, ValueError) as e:
            if self.verbose:
                _log(f"Open failed on {target}: {e}")
            return False

        with self._write_lock:
            self.serial_conn = ser
            self.port = target
            self._need_reopen = False
        self.last_rx_time = time.time()
        return True

    def _close(self):
        with self._write_lock:
            ser, self.serial_conn = self.serial_conn, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    @property
    def is_connected(self) -> bool:
        return self.serial_conn is not None and not self._need_reopen

    @property
    def link_ok(self) -> bool:
        """Port open AND the ESP32 has said something recently (heartbeat every 500 ms)."""
        return self.is_connected and (time.time() - self.last_rx_time) < self.HEARTBEAT_TIMEOUT

    # ------------------------------------------------------------------ public API
    def connect(self, wait: float = 3.0) -> bool:
        """Opens the port and starts the background IO thread. Waits up to `wait` s for the port."""
        if self._running and self.is_connected:
            return True
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._io_loop, name="wro-serial", daemon=True)
            self._thread.start()

        deadline = time.time() + wait
        while time.time() < deadline:
            if self.is_connected:
                _log(f"Connected to ESP32 on {self.port}")
                return True
            time.sleep(0.05)
        _log("ESP32 serial port not found. Check the USB cable (the link will keep retrying).")
        return False

    def disconnect(self):
        """Stops the motor, stops the IO thread and closes the port."""
        if self.is_connected:
            self._write_raw(b"STOP\n")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._close()
        _log(f"Serial port closed. Link stats this run: {self.stats}")

    def send_command(self, command: str) -> bool:
        """
        Sends one command. Returns immediately: False if the command was invalid or the
        link is down (the ESP32 failsafe keeps the car safe), True if it was written.
        Supports: FORWARD, BACKWARD, LEFT, RIGHT, STOP, TURN_LEFT, TURN_RIGHT, AUTO_US_ON/OFF,
                  STEER:<angle>, DRIVE:<speed>:<angle>, SET_SPEED:<pwm>, SET_TURN_DELAY:<ms>
        """
        cmd = command.strip().upper()
        if not self._is_valid(cmd):
            print(f"[ERROR] Rejected invalid command '{command}'.", file=sys.stderr)
            return False
        return self._write_raw(f"{cmd}\n".encode("ascii"))

    def send_steer(self, angle: int) -> bool:
        return self.send_command(f"STEER:{int(angle)}")

    def send_drive(self, speed: int, angle: int) -> bool:
        return self.send_command(f"DRIVE:{int(speed)}:{int(angle)}")

    def get_us_data(self) -> dict:
        """Kept for older scripts. This build has no ultrasonic sensors, so values stay 0."""
        return self.us_data

    # ------------------------------------------------------------------ internals
    def _is_valid(self, cmd: str) -> bool:
        if cmd in self.VALID_COMMANDS:
            return True
        parts = cmd.split(":")

        def is_int(s):
            return s.lstrip("-").isdigit()

        if parts[0] in ("STEER", "SET_SPEED", "SET_TURN_DELAY") and len(parts) == 2 and is_int(parts[1]):
            return True
        if parts[0] == "DRIVE" and len(parts) == 3 and is_int(parts[1]) and is_int(parts[2]):
            return True
        return False

    def _write_raw(self, data: bytes) -> bool:
        with self._write_lock:
            ser = self.serial_conn
            if ser is None or self._need_reopen:
                return False
            try:
                ser.write(data)
                return True
            except serial.SerialTimeoutException:
                self.stats["dropped_writes"] += 1
                return False
            except (serial.SerialException, OSError) as e:
                if not self._need_reopen:
                    _log(f"Write failed ({e}). Reconnecting in background...")
                self._need_reopen = True
                return False

    def _io_loop(self):
        buf = b""
        next_retry = 0.0
        was_connected = False
        while self._running:
            if self.serial_conn is None or self._need_reopen:
                if was_connected:
                    self.stats["disconnects"] += 1
                    _log(f"LOST serial port {self.port}. Motor is stopped by the ESP32 failsafe. Retrying...")
                    was_connected = False
                self._close()
                buf = b""
                if time.time() >= next_retry:
                    if self._open():
                        if self.stats["disconnects"]:
                            self.stats["reconnects"] += 1
                            _log(f"RECONNECTED on {self.port}")
                        was_connected = True
                    else:
                        next_retry = time.time() + 0.25
                time.sleep(0.02)
                continue

            was_connected = True
            ser = self.serial_conn
            try:
                chunk = ser.read(ser.in_waiting or 1)
            except (serial.SerialException, OSError, TypeError, AttributeError) as e:
                if self._running:
                    _log(f"Read failed ({e})")
                self._need_reopen = True
                continue

            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("ascii", errors="ignore").strip()
                if line:
                    self._handle_line(line)
            if len(buf) > 512:  # garbage without newlines
                buf = b""

    def _handle_line(self, line: str):
        self.last_rx_time = time.time()
        self.last_ack = line

        if line.startswith("HB:"):
            try:
                uptime = int(line[3:])
            except ValueError:
                return
            if self._last_uptime_ms is not None and uptime + 200 < self._last_uptime_ms:
                self.stats["esp_reboots"] += 1
                _log(f"ESP32 REBOOTED (uptime {self._last_uptime_ms} -> {uptime} ms). "
                     f"Likely a power dip / brownout on the ESP32 supply.")
            self._last_uptime_ms = uptime
            return

        if line.startswith("BOOT:"):
            reason = line.split(":")[-1]
            self.stats["last_reset_reason"] = reason
            if self._last_uptime_ms is not None:
                self.stats["esp_reboots"] += 1
            self._last_uptime_ms = 0
            hint = " <-- POWER PROBLEM (supply voltage dipped)" if "BROWNOUT" in reason else ""
            _log(f"ESP32 booted, reset reason: {reason}{hint}")
            return

        if line.startswith("INFO:FAILSAFE_STOP"):
            self.stats["failsafe_stops"] += 1
            _log("ESP32 failsafe stop: no command received for 500 ms (Pi loop stalled or link down)")
            return

        if line.startswith("US:"):
            self.stats["us_lines"] += 1
            try:
                for part in line[3:].split(","):
                    k, v = part.split(":")
                    self.us_data[k.strip().lower()] = int(v)
            except ValueError:
                pass
            return

        if self.on_ack_callback:
            self.on_ack_callback(line)
        elif line.startswith(("ERR", "ERROR")) or self.verbose:
            _log(f"ESP32 >> {line}")


class Drive:
    """
    Sends DRIVE commands only when speed/angle change, plus a keepalive so the ESP32
    500 ms failsafe never fires during normal driving.
    """

    def __init__(self, link: WROSerialController, keepalive: float = 0.1):
        self.link = link
        self.keepalive = keepalive
        self.speed = None
        self.angle = None
        self._last_send = 0.0

    def drive(self, speed: int, angle: int, force: bool = False):
        speed, angle = int(speed), int(angle)
        now = time.time()
        if force or speed != self.speed or angle != self.angle or now - self._last_send >= self.keepalive:
            self.link.send_command(f"DRIVE:{speed}:{angle}")
            self.speed, self.angle, self._last_send = speed, angle, now

    def hold(self, speed: int, angle: int, seconds: float, tick: Optional[Callable[[], object]] = None):
        """Drives at a fixed speed/angle for `seconds`, re-sending so the failsafe stays satisfied."""
        end = time.time() + seconds
        while time.time() < end:
            self.drive(speed, angle)
            if tick is not None:
                tick()
            else:
                time.sleep(0.02)

    def stop(self, brake_speed: int = 0, center: int = 100):
        """Stops the car. A negative brake_speed gives a short reverse pulse first to kill momentum."""
        if brake_speed:
            self.link.send_command(f"DRIVE:{int(brake_speed)}:{center}")
            time.sleep(0.08)
        self.link.send_command("STOP")
        self.speed, self.angle = 0, center


_default_controller: Optional[WROSerialController] = None


def get_controller(port: Optional[str] = None) -> WROSerialController:
    global _default_controller
    if _default_controller is None:
        _default_controller = WROSerialController(port=port)
    return _default_controller


def send_command(command: str) -> bool:
    return get_controller().send_command(command)


if __name__ == "__main__":
    print("WRO serial link self-test (Ctrl+C to quit). Robot should be elevated.")
    ctrl = WROSerialController(verbose=True)
    try:
        for _ in range(10):
            ctrl.send_command("FORWARD")
            time.sleep(0.1)
        ctrl.send_command("STOP")
        print("Watching link for 10 s (heartbeats every 0.5 s)...")
        t_end = time.time() + 10
        while time.time() < t_end:
            print(f"  link_ok={ctrl.link_ok} port={ctrl.port} stats={ctrl.stats}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.disconnect()
