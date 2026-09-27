#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Branch: camera-only-canada-strategy  (this file is a bench tool, same on every branch)

Does the motor actually REVERSE?

Why this exists
  The car stops dead in front of a traffic sign and never backs away, and only starts
  again after the ESP32 is restarted. Two very different causes look identical from the
  driver's seat:

    A. Reverse is not driving at all. The firmware writes PWM to the second pin
       (IN2 / LPWM); if that pin is not connected, or the driver's second half is not
       enabled, "reverse" only takes the forward duty to zero. That is a STOP, not a
       reverse. The car then sits on the sign at full forward throttle, stalls, drags the
       supply down and takes the ESP32 with it - which is why it needs a restart.

    B. Reverse works, but the current it draws browns the board out.

  This tells them apart. It drives forward, stops, reverses, stops, and reports what the
  ESP32 said during each phase - telemetry lines, failsafe stops, and any reboot with its
  reset reason. BROWNOUT during the reverse phase means B. Silence from the wheels with a
  perfectly healthy link means A.

WHEELS OFF THE GROUND. Put the car on a stand or hold it up. It will drive at full speed.

Usage
  python3 test_reverse.py                 forward/reverse at the speed the run code uses
  python3 test_reverse.py --speed 200     at some other duty
  python3 test_reverse.py --ramp          step 150..255 both ways, find where reverse bites
  python3 test_reverse.py --seconds 3     longer phases

What to watch
  The wheel, and nothing else. For each phase say out loud which way it turns.
    forward spins, reverse dead silent ............ cause A: the reverse half of the
                                                    driver is not driving. Check IN2/LPWM
                                                    continuity from ESP32 GPIO 33, and on a
                                                    BTS7960 that L_EN is actually high.
    forward spins, reverse twitches then stops .... cause A or a current limit: a DRV8833
                                                    is only good for ~1.5 A a channel and
                                                    trips instantly on a stalled 370 motor.
    both spin, but the link drops or the board
    reboots with BROWNOUT ........................ cause B: power. Bulk capacitor across
                                                    the motor supply, separate the logic
                                                    rail, thicker wires.
    both spin cleanly, no reboots ................ the driver is fine; the freeze is in
                                                    the vision thresholds instead. Run
                                                    tune_rois.py --round 2 and compare the
                                                    bumper box reading with
                                                    BUMPER_BLOCK_AREA.
"""

import argparse
import sys
import time

from wro_serial import WROSerialController

SERVO_CENTER = 100


def parse_args():
    p = argparse.ArgumentParser(description="Check that the motor really reverses")
    p.add_argument("--speed", type=int, default=230, help="PWM duty to test (0-255)")
    p.add_argument("--seconds", type=float, default=2.0, help="seconds per phase")
    p.add_argument("--ramp", action="store_true", help="step through several duties instead")
    p.add_argument("--port", default=None, help="serial port (default: auto-detect)")
    p.add_argument("--yes", action="store_true", help="skip the wheels-off-the-ground prompt")
    return p.parse_args()


class Phase:
    """One driving phase, with the ESP32's side of the story recorded around it."""

    def __init__(self, link, label, speed, seconds):
        self.link, self.label, self.speed, self.seconds = link, label, speed, seconds

    def run(self):
        before = dict(self.link.stats)
        print(f"\n  {self.label:28s} DRIVE:{self.speed}:{SERVO_CENTER}  for {self.seconds:.1f}s")
        end = time.time() + self.seconds
        sent = 0
        while time.time() < end:
            # resend continuously: the firmware stops the motor if nothing arrives for 500 ms
            if self.link.send_command(f"DRIVE:{self.speed}:{SERVO_CENTER}"):
                sent += 1
            time.sleep(0.04)
        after = dict(self.link.stats)
        d = {k: v - before.get(k, 0) for k, v in after.items() if isinstance(v, int)}
        note = []
        if d.get("esp_reboots"):
            note.append(f"REBOOTED {d['esp_reboots']}x (reason: {after.get('last_reset_reason')})")
        if d.get("disconnects"):
            note.append(f"link dropped {d['disconnects']}x")
        if d.get("failsafe_stops"):
            note.append(f"failsafe stopped the motor {d['failsafe_stops']}x")
        if d.get("reset_pulses"):
            note.append(f"the Pi had to reboot the board {d['reset_pulses']}x")
        if d.get("dropped_writes"):
            note.append(f"{d['dropped_writes']} commands could not be sent")
        print(f"  {'':28s} sent {sent}, telemetry {d.get('us_lines', 0)} lines"
              + ("  ** " + "; ".join(note) + " **" if note else "   link healthy"))
        return note


def main():
    args = parse_args()
    print(__doc__.split("Usage")[0].rstrip())

    if not args.yes:
        print("\n  WHEELS OFF THE GROUND. The car will drive at full speed.")
        try:
            if input("  Ready? [y/N] ").strip().lower() not in ("y", "yes"):
                print("  Cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 0

    link = WROSerialController(port=args.port, auto_connect=False)
    if not link.connect(wait=5.0):
        print("\n[ERROR] No ESP32 found. Check the USB cable and that the firmware is flashed.")
        return 1
    time.sleep(0.6)                      # let the first telemetry and any BOOT: line arrive
    print(f"[LINK] Connected on {link.port}. Reset reason: {link.stats['last_reset_reason']}")

    duties = ([150, 180, 200, 220, 235, 255] if args.ramp else [args.speed])
    trouble = []
    try:
        for duty in duties:
            print(f"\n=== duty {duty} " + "=" * 40)
            for label, spd in ((f"FORWARD  {duty}", duty), ("stop", 0),
                               (f"REVERSE  -{duty}", -duty), ("stop", 0)):
                trouble += Phase(link, label, spd, 0.4 if spd == 0 else args.seconds).run()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted.")
    finally:
        for _ in range(10):
            link.send_command(f"DRIVE:0:{SERVO_CENTER}")
            time.sleep(0.02)
        time.sleep(0.2)
        print(f"\n[STATS] {link.stats}")
        link.disconnect()

    print("\n" + "=" * 62)
    if trouble:
        print("The link or the board had trouble during the test - see the ** notes ** above.")
        print("A reboot or a brownout while reversing is cause B: the power supply.")
    else:
        print("The link stayed healthy throughout, so whatever the wheel did was what the")
        print("driver was told to do. If the wheel did not turn backwards, the reverse half")
        print("of the driver is not driving - cause A, and it is wiring, not code.")
    print("Read the 'What to watch' table at the top of this file for what to do next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
