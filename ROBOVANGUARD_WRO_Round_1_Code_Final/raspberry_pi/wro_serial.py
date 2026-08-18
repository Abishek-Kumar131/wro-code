#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2025
Raspberry Pi 5 to ESP32 USB Serial Communication Module

Features:
- USB Serial at 115200 baud
- Automatic port detection (/dev/ttyUSB*, /dev/ttyACM*, COM*)
- Non-blocking ACK reading & response logging
- Auto-reconnection and disconnection fault-tolerance
- Thread-safe command dispatching
"""

import time
import threading
import sys
import glob
from typing import Optional, Callable

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] 'pyserial' is not installed. Please install it using: pip install pyserial", file=sys.stderr)


class WROSerialController:
    """Manages physical USB serial communication between Raspberry Pi 5 and ESP32."""

    VALID_COMMANDS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.1, auto_connect: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None
        self.is_running = False
        self._lock = threading.Lock()
        self._rx_thread: Optional[threading.Thread] = None
        self.last_ack: Optional[str] = None
        self.on_ack_callback: Optional[Callable[[str], None]] = None

        if auto_connect:
            self.connect()

    def find_serial_port(self) -> Optional[str]:
        """Automatically detects standard ESP32 USB Serial port across Linux and Windows."""
        if self.port and self.port != "AUTO":
            return self.port

        # Search using serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # Common USB-UART bridge chips (CP210x, CH340, FTDI, ESP32-S3 USB CDC)
            desc = p.description.lower()
            if any(k in desc for k in ["cp210", "ch340", "ftdi", "usb serial", "uart", "esp32", "acm"]):
                print(f"[INFO] Auto-detected ESP32 Serial Port: {p.device} ({p.description})")
                return p.device

        # Fallback search for Linux device nodes
        linux_candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if linux_candidates:
            print(f"[INFO] Auto-detected Linux USB Serial Port: {linux_candidates[0]}")
            return linux_candidates[0]

        if ports:
            print(f"[INFO] Defaulting to first available port: {ports[0].device}")
            return ports[0].device

        return None

    def connect(self) -> bool:
        """Establishes USB Serial connection to ESP32."""
        with self._lock:
            if self.serial_conn and self.serial_conn.is_open:
                return True

            target_port = self.find_serial_port()
            if not target_port:
                print("[WARNING] No USB serial port found for ESP32. Retrying on demand.", file=sys.stderr)
                return False

            try:
                print(f"[INFO] Connecting to ESP32 on {target_port} at {self.baudrate} baud...")
                self.serial_conn = serial.Serial(
                    port=target_port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=0.2
                )
                self.port = target_port
                # Allow ESP32 to reset if DTR toggles
                time.sleep(1.0)
                self.serial_conn.reset_input_buffer()
                self.serial_conn.reset_output_buffer()

                self.is_running = True
                self._rx_thread = threading.Thread(target=self._read_loop, daemon=True)
                self._rx_thread.start()

                print(f"[SUCCESS] Connected to ESP32 on {target_port}!")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to connect to {target_port}: {e}", file=sys.stderr)
                self.serial_conn = None
                return False

    def disconnect(self):
        """Safely closes the serial connection."""
        self.is_running = False
        with self._lock:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    # Send final STOP command before closing
                    self.serial_conn.write(b"STOP\n")
                    self.serial_conn.flush()
                    self.serial_conn.close()
                except Exception:
                    pass
                self.serial_conn = None
        print("[INFO] USB Serial disconnected.")

    def _read_loop(self):
        """Background thread for asynchronous reading of ACKs and telemetry."""
        while self.is_running:
            try:
                if self.serial_conn and self.serial_conn.is_open:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode("utf-8", errors="replace").strip()
                        if line:
                            self.last_ack = line
                            if self.on_ack_callback:
                                self.on_ack_callback(line)
                            else:
                                print(f"[ESP32 >> RPi5]: {line}")
                time.sleep(0.01)
            except Exception as e:
                print(f"[WARNING] Serial read error: {e}", file=sys.stderr)
                time.sleep(0.5)

    def send_command(self, command: str) -> bool:
        """
        Sends a movement command to the ESP32.
        Command is automatically capitalized and formatted with a trailing newline.
        """
        cmd_clean = command.strip().upper()
        if cmd_clean not in self.VALID_COMMANDS:
            print(f"[ERROR] Invalid command '{command}'. Valid commands are: {self.VALID_COMMANDS}", file=sys.stderr)
            return False

        message = f"{cmd_clean}\n".encode("utf-8")

        with self._lock:
            if not self.serial_conn or not self.serial_conn.is_open:
                # Attempt silent reconnection
                if not self.connect():
                    return False

            try:
                self.serial_conn.write(message)
                self.serial_conn.flush()
                return True
            except (serial.SerialException, OSError) as e:
                print(f"[ERROR] Failed to transmit command '{cmd_clean}' over USB: {e}", file=sys.stderr)
                if self.serial_conn:
                    try:
                        self.serial_conn.close()
                    except Exception:
                        pass
                    self.serial_conn = None
                return False


# Global singleton controller instance for simple functional access
_default_controller: Optional[WROSerialController] = None


def get_controller(port: Optional[str] = None) -> WROSerialController:
    """Returns or initializes the global serial controller singleton."""
    global _default_controller
    if _default_controller is None:
        _default_controller = WROSerialController(port=port)
    return _default_controller


def send_command(command: str) -> bool:
    """
    Convenience function for direct invocation:
    send_command("FORWARD")
    send_command("LEFT")
    send_command("RIGHT")
    send_command("BACKWARD")
    send_command("STOP")
    """
    controller = get_controller()
    return controller.send_command(command)


if __name__ == "__main__":
    print("Testing WRO Serial Module standalone...")
    ctrl = WROSerialController()
    if ctrl.connect():
        print("Sending test command: FORWARD")
        ctrl.send_command("FORWARD")
        time.sleep(1.0)
        print("Sending test command: STOP")
        ctrl.send_command("STOP")
        time.sleep(0.5)
        ctrl.disconnect()
    else:
        print("Could not connect to ESP32.")
