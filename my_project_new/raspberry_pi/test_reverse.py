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
  python3 test_reverse.py --pins          drive each motor pin on its own, to find which
                                          half is dead (needs the PINTEST firmware)
  python3 test_reverse.py --from-motion   reverse straight out of forward motion, with no
                                          stop first - what the R1 manoeuvre really does
  python3 test_reverse.py --kick 150      break away from a standstill at full duty first

Check the wiring FIRST, before running anything
  A plain BO / brushed motor needs no reverse pin: reversing it means swapping plus and
  minus, and that is the H-bridge's whole job. But it can only swap them if BOTH motor
  leads land on the driver's two outputs - OUT1 and OUT2 of one channel (AOUT1/AOUT2 on a
  DRV8833, M+/M- on a BTS7960). If one lead sits on GND or battery negative instead:

      command                       OUT1   OUT2   OUT1+OUT2      OUT1+GND
      forward (IN1=PWM, IN2=0)      PWM    low    spins          spins
      reverse (IN1=0,   IN2=PWM)    low    PWM    spins back     NOTHING (0 V across it)

  That single wire explains forward working, reverse doing nothing, and the link being
  perfectly healthy while it happens. Follow the motor's two leads with your eyes: if
  either one goes anywhere except an output terminal of the driver, that is the fault.

What to watch
  The wheel, and nothing else. For each phase say out loud which way it turns.
    forward spins, reverse dead silent ............ cause A: nothing is driving the motor
                                                    the other way. In order of likelihood:
                                                    a motor lead on GND instead of OUT2
                                                    (see above); the IN2/LPWM jumper from
                                                    ESP32 GPIO 33 not making contact; L_EN
                                                    low on a BTS7960. Use --pins to tell
                                                    the pin apart from the output.
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
    p.add_argument("--pins", action="store_true",
                   help="drive each motor pin on its own (needs the PINTEST firmware)")
    p.add_argument("--from-motion", dest="from_motion", action="store_true",
                   help="reverse straight out of forward motion, with no stop in between")
    p.add_argument("--kick", type=int, default=0, metavar="MS",
                   help="start the reverse at full duty for this many ms to break away")
    p.add_argument("--port", default=None, help="serial port (default: auto-detect)")
    p.add_argument("--yes", action="store_true", help="skip the wheels-off-the-ground prompt")
    return p.parse_args()


class Phase:
    """One driving phase, with the ESP32's side of the story recorded around it."""

    def __init__(self, link, label, speed, seconds, command=None):
        self.link, self.label, self.speed, self.seconds = link, label, speed, seconds
        self.command = command or f"DRIVE:{speed}:{SERVO_CENTER}"
        self.kick_ms = getattr(Phase, "kick_default", 0)

    def run(self):
        before = dict(self.link.stats)
        print(f"\n  {self.label:28s} {self.command}  for {self.seconds:.1f}s")
        end = time.time() + self.seconds
        sent = 0
        if self.kick_ms and self.speed < 0:
            print(f"  {'':28s} breaking away at full duty for {self.kick_ms} ms first")
            kick_end = time.time() + self.kick_ms / 1000.0
            while time.time() < kick_end:
                self.link.send_command(f"DRIVE:-255:{SERVO_CENTER}")
                time.sleep(0.02)
        while time.time() < end:
            # resend continuously: the firmware stops the motor if nothing arrives for 500 ms
            if self.link.send_command(self.command):
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

    Phase.kick_default = args.kick     # applies to every reverse phase below
    trouble = []
    if args.from_motion:
        # Reverse with no stop phase in between. If this moves the car and the plain test
        # does not, the drive is wired correctly and the problem is breaking away from a
        # standstill - try --kick, and see REVERSE_KICK_TIME in obstacle_challenge_R2.py.
        print(f"\n=== reverse straight out of forward motion " + "=" * 18)
        try:
            trouble += Phase(link, f"FORWARD  {args.speed}", args.speed, args.seconds).run()
            trouble += Phase(link, f"REVERSE  -{args.speed} (no stop)", -args.speed,
                             args.seconds).run()
        except KeyboardInterrupt:
            print(f"\n[STOP] Interrupted.")
        finally:
            for _ in range(10):
                link.send_command(f"DRIVE:0:{SERVO_CENTER}")
                time.sleep(0.02)
            time.sleep(0.2)
            print(f"\n[STATS] {link.stats}")
            link.disconnect()
        print(f"\n" + "=" * 62)
        print("  it reversed this time, but not from a standstill .. the wiring is fine. The")
        print("      motor cannot break away backwards from rest. Use --kick 150 to confirm,")
        print("      and REVERSE_KICK_TIME in obstacle_challenge_R2.py does this during a run.")
        print("  it did not reverse either way ..................... back to --pins: nothing")
        print("      is driving the motor the other way at all.")
        return 0

    if args.pins:
        # Isolate the two halves. motor_forward/backward always write both pins, so they
        # cannot tell a dead GPIO from a dead driver channel; PINTEST drives one at a time.
        print(f"\n=== one pin at a time " + "=" * 32)
        print("  PIN 1 is ESP32 GPIO 32 -> IN1/RPWM.  PIN 2 is GPIO 33 -> IN2/LPWM.")
        print("  Each should spin the wheel, in opposite directions. Note which ones do.")
        try:
            for which in (1, 2):
                trouble += Phase(link, f"PIN {which}  duty {args.speed}", args.speed,
                                 args.seconds, command=f"PINTEST:{which}:{args.speed}").run()
                trouble += Phase(link, "stop", 0, 0.4, command="PINTEST:1:0").run()
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
        print("  both pins spin the wheel ..... the drive hardware is fine both ways, and the")
        print("                                fault is in how the run code commands it")
        print("  only PIN 1 spins it ......... the motor is only ever driven one way. Check,")
        print("                                in this order: (1) both motor leads land on")
        print("                                the driver OUTPUTS - if one is on GND or")
        print("                                battery negative, no H-bridge can reverse it;")
        print("                                (2) the IN2/LPWM jumper from GPIO 33 makes")
        print("                                contact at both ends; (3) L_EN is high on a")
        print("                                BTS7960. If GPIO 33 itself is dead, move that")
        print("                                wire to a free pin and set motorPin2 to it -")
        print("                                GPIO 21 and 22 are free on this build.")
        print("  only PIN 2 spins it ......... the same story on the GPIO 32 / IN1 side")
        print("  neither spins it ............ the driver has no motor supply, or nSLEEP")
        print("                                (GPIO 13) is not high, or the motor leads are")
        print("                                off. Forward worked before, so suspect the")
        print("                                leads or the supply first.")
        return 0

    duties = ([150, 180, 200, 220, 235, 255] if args.ramp else [args.speed])
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
