#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 competition launcher (run at boot by wro_autostart.service).

Starts the challenge script immediately. The script loads OpenCV, opens the camera and the
ESP32 link, and THEN waits for the start button - so pressing the button starts the car
within about half a second instead of after several seconds of start-up.

  competition_launcher.py --r1 [args]   Open Challenge   (open_challenge_R1.py)
  competition_launcher.py --r2 [args]   Obstacle Challenge (obstacle_challenge_R2.py, default)

Every other argument (--pin, --active-high, --no-display, --webcam, --dir ...) is passed
through to the challenge script.
"""

import os
import subprocess
import sys


def main():
    target = "open_challenge_R1.py" if "--r1" in sys.argv else "obstacle_challenge_R2.py"
    passthrough = [a for a in sys.argv[1:] if a not in ("--r1", "--r2")]

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), target)
    cmd = [sys.executable, script] + passthrough
    print(f"[LAUNCHER] Starting: {' '.join(cmd)}", flush=True)

    try:
        sys.exit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        print("\n[LAUNCHER] Stopped by user.")


if __name__ == "__main__":
    main()
