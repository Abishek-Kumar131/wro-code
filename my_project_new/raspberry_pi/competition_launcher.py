#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Competition Launcher Script

Runs on Pi 5 boot:
- Waits for physical push button press on GPIO 17.
- Launches Round 1 Open Challenge code (open_challenge_R1.py) automatically!
"""

import sys
import os
import time
import subprocess
import select

def wait_for_button(gpio_pin=17):
    print("=" * 65)
    print("   ROBOVANGUARD WRO 2026 COMPETITION LAUNCHER")
    print(f"   Waiting for physical push button press on GPIO {gpio_pin}...")
    print("   (Or press ENTER in terminal)")
    print("=" * 65)

    button_obj = None
    try:
        from gpiozero import Button
        button_obj = Button(gpio_pin, pull_up=True)
        print(f"[GPIO] gpiozero Button initialized on GPIO {gpio_pin}.")
    except Exception:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            print(f"[GPIO] RPi.GPIO initialized on GPIO {gpio_pin}.")
        except Exception as e:
            print(f"[GPIO INFO] GPIO library fallback ({e}). Keyboard trigger active.")

    while True:
        if button_obj is not None and button_obj.is_pressed:
            print("\n[LAUNCH] PHYSICAL BUTTON PRESSED! Starting Round 1 Open Challenge...")
            return True

        if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
            sys.stdin.readline()
            print("\n[LAUNCH] ENTER key pressed! Starting Round 1 Open Challenge...")
            return True

        time.sleep(0.05)

def main():
    gpio_pin = 17
    if "--pin" in sys.argv:
        idx = sys.argv.index("--pin")
        if idx + 1 < len(sys.argv):
            gpio_pin = int(sys.argv[idx + 1])

    wait_for_button(gpio_pin)

    # Path to open_challenge_R1.py script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    r1_script = os.path.join(script_dir, "open_challenge_R1.py")

    # Forward any arguments like --no-display or --webcam
    extra_args = [arg for arg in sys.argv[1:] if arg not in ("--pin", str(gpio_pin))]
    cmd = [sys.executable, r1_script, "--no-wait"] + extra_args
    print(f"[EXEC] Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Stopped by user.")

if __name__ == "__main__":
    main()
