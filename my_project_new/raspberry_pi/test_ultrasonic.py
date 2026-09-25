#!/usr/bin/env python3
"""
BRANCH: us-vision-hybrid  (camera + 6 ultrasonic sensors)

ROBOVANGUARD - WRO Future Engineers 2026
Ultrasonic sensor test and diagnostic tool.

Shows every sensor live and, when you stop it, prints a verdict per sensor so you can
tell a dead sensor from a noisy one. Nothing moves: the motor is never commanded.

Reminder on what the numbers mean:
  0  = no echo came back within range. That means "nothing in front of me", NOT "0 cm".
       A sensor pointing down an open track reads 0 and that is correct.
  The firmware pings ONE sensor every 20 ms in rotation, so each sensor refreshes
  about every 120 ms (roughly 8 updates per second).

Usage
  python3 test_ultrasonic.py                    live table of all six sensors
  python3 test_ultrasonic.py --seconds 20       run 20 s, then print the summary
  python3 test_ultrasonic.py --sensor f         watch one sensor only
  python3 test_ultrasonic.py --sensor f --expect 30
                                                target placed at 30 cm: show the error
  python3 test_ultrasonic.py --csv log.csv      also log every sample to CSV
  python3 test_ultrasonic.py --port /dev/ttyUSB0

How to use it on the bench
  1. Run it with nothing in front of the car. Every sensor should read 0.
     A sensor showing a small number with nothing in front of it is picking up the
     chassis or the floor: re-aim it.
  2. Hold a flat object (a book) about 30 cm from one sensor at a time and check that
     only that sensor responds, and that the value is within a couple of cm of a tape
     measure. Use --sensor f --expect 30 to get the error printed for you.
  3. Watch the jitter column. Under ~2 cm on a still target is healthy. Large jitter
     with the motor running usually means vibration or sensors hearing each other.
"""

import argparse
import statistics
import sys
import time

from wro_serial import WROSerialController

KEYS = ("f", "f1", "f2", "l", "r", "b")
LABELS = {"f": "front", "f1": "front-left", "f2": "front-right",
          "l": "left", "r": "right", "b": "back"}


def parse_args():
    p = argparse.ArgumentParser(description="Ultrasonic sensor test (no motor movement)")
    p.add_argument("--seconds", type=float, default=0, help="run this long, then summarise (0 = until Ctrl+C)")
    p.add_argument("--sensor", choices=KEYS, help="watch a single sensor")
    p.add_argument("--expect", type=float, help="known target distance in cm; prints the error")
    p.add_argument("--csv", help="write every sample to this CSV file")
    p.add_argument("--port", help="serial port (default: auto-detect)")
    p.add_argument("--rate", type=float, default=10.0, help="samples per second (default 10)")
    return p.parse_args()


class SensorStats:
    def __init__(self, key):
        self.key = key
        self.samples = 0
        self.zeros = 0
        self.values = []      # non-zero readings only
        self.last = 0

    def add(self, value):
        self.samples += 1
        self.last = value
        if value == 0:
            self.zeros += 1
        else:
            self.values.append(value)

    @property
    def zero_pct(self):
        return 100.0 * self.zeros / self.samples if self.samples else 0.0

    @property
    def mean(self):
        return statistics.fmean(self.values) if self.values else 0.0

    @property
    def jitter(self):
        return statistics.pstdev(self.values) if len(self.values) > 1 else 0.0

    def row(self):
        if not self.values:
            return (f"{LABELS[self.key]:>11s} ({self.key:2s})  {self.last:4d}      -     -     -"
                    f"      -    {self.zero_pct:5.1f}%")
        return (f"{LABELS[self.key]:>11s} ({self.key:2s})  {self.last:4d}  {min(self.values):4d}  "
                f"{max(self.values):4d}  {self.mean:5.1f}  {self.jitter:5.2f}  {self.zero_pct:5.1f}%")

    def verdict(self, expect=None):
        if self.samples == 0:
            return "NO DATA", "no samples were taken"
        if not self.values:
            return "SILENT", ("never returned a reading. Either nothing was in range (fine if the "
                              "track was clear) or the sensor is dead / mis-wired / the echo pin is wrong")
        if self.zero_pct > 60:
            return "PATCHY", (f"{self.zero_pct:.0f}% of samples had no echo. Weak return: check the aim, "
                              f"the target may be angled or too far")
        if self.zero_pct > 20:
            return "DROPOUTS", (f"{self.zero_pct:.0f}% of samples had no echo. Works, but loses the target "
                                f"often - check the aim and that the target is flat on")
        if max(self.values) == min(self.values) and len(self.values) >= 30:
            return "CHECK", (f"every reading was exactly {self.last} cm. Normal if the target really is "
                             f"fixed at that distance - move the target and re-run to confirm it tracks")
        if self.jitter > 5:
            return "NOISY", (f"jitter {self.jitter:.1f} cm. Usually vibration, a slanted target, or two "
                             f"sensors hearing each other")
        if expect is not None:
            err = self.mean - expect
            if abs(err) > 3:
                return "OFF", f"average {self.mean:.1f} cm vs {expect:.0f} cm expected (error {err:+.1f} cm)"
            return "GOOD", f"average {self.mean:.1f} cm vs {expect:.0f} cm expected (error {err:+.1f} cm)"
        return "GOOD", f"average {self.mean:.1f} cm, jitter {self.jitter:.2f} cm"


def main():
    args = parse_args()
    keys = (args.sensor,) if args.sensor else KEYS

    print("=" * 74)
    print("   ROBOVANGUARD - Ultrasonic sensor test   (branch: us-vision-hybrid)")
    print("   The motor is never commanded. Ctrl+C to stop and see the summary.")
    print("=" * 74)

    link = WROSerialController(port=args.port, auto_connect=False)
    if not link.connect(wait=5.0):
        print("\n[ERROR] No ESP32 found. Check the USB cable, then try --port /dev/ttyUSB0")
        return 1

    stats = {k: SensorStats(k) for k in keys}
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", encoding="utf-8")
        csv_file.write("time_s," + ",".join(keys) + "\n")

    header = ("     sensor        now   min   max   mean  jitter  no-echo\n"
              "  " + "-" * 60)
    printed_lines = 0
    interactive = sys.stdout.isatty()
    last_dump = 0.0
    t_start = time.time()
    warned_no_telemetry = False

    try:
        while True:
            now = time.time()
            if args.seconds and now - t_start >= args.seconds:
                break

            data = link.get_us_data()
            for k in keys:
                try:
                    stats[k].add(int(data.get(k, 0) or 0))
                except (TypeError, ValueError):
                    stats[k].add(0)

            if csv_file:
                csv_file.write(f"{now - t_start:.2f}," + ",".join(str(stats[k].last) for k in keys) + "\n")

            # ---- live view: redrawn in place on a terminal, throttled when piped to a file
            lines = [header] + ["  " + stats[k].row() for k in keys]
            lines.append(f"  elapsed {now - t_start:5.1f}s   telemetry lines {link.stats['us_lines']:5d}   "
                         f"link {'ok' if link.link_ok else 'DOWN'}")
            if interactive:
                if printed_lines:
                    sys.stdout.write(f"\033[{printed_lines}A")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
                printed_lines = len(lines)
            elif now - last_dump >= 2.0:
                last_dump = now
                print("\n".join(lines), flush=True)

            if (not warned_no_telemetry and now - t_start > 3.0 and link.stats["us_lines"] == 0):
                warned_no_telemetry = True
                print("\n[WARNING] The ESP32 has not sent a single US: line in 3 s.")
                print("          This branch's firmware should send them 10x per second.")
                print("          Re-flash the firmware from this branch, then run this again.\n")
                printed_lines = 0

            time.sleep(max(0.01, 1.0 / args.rate))

    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()
            print(f"\n[CSV] Wrote {args.csv}")

        print("\n" + "=" * 74)
        print("   SUMMARY")
        print("=" * 74)
        if link.stats["us_lines"] == 0:
            print("  No ultrasonic telemetry was received at all.")
            print("  The firmware on the ESP32 is not sending US: lines - re-flash this branch's")
            print("  firmware. Nothing below is meaningful until that is fixed.")
        for k in keys:
            tag, why = stats[k].verdict(args.expect)
            print(f"  {LABELS[k]:>11s} ({k:2s})  {tag:7s}  {why}")
        print()
        print(f"  Samples per sensor: {stats[keys[0]].samples} over {time.time() - t_start:.1f}s")
        print(f"  Link stats: {link.stats}")
        print()
        print("  Reminder: 0 means 'nothing in range', not '0 cm'. With a clear track in front")
        print("  of the car, SILENT on the front sensors is the correct result.")
        link.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
