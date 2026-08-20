#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Architecture:
- Raspberry Pi 5 runs OpenCV vision with Picamera2 (Pi Cam v2)
- Computes PD wall centering & detects Orange/Blue corner markers
- Streams commands over USB Serial to ESP32 controller (115200 baud)
- ESP32 controls DC motor & steering servo with 500ms watchdog failsafe
"""

import sys
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from wro_serial import WROSerialController
from masks import rOrange, rBlack, rBlue
from wro_functions import find_contours, max_contour, display_roi, display_variables


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 1 Open Challenge Node (Pi 5)")
    print("=" * 65)

    # Initialize USB Serial connection to ESP32
    serial_ctrl = WROSerialController()
    if not serial_ctrl.connect():
        print("[ERROR] Cannot proceed without ESP32 serial connection.")
        sys.exit(1)

    time.sleep(1.0)

    # Initialize Pi Camera v2 via Picamera2
    print("[INFO] Initializing Picamera2...")
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.controls.FrameRate = 30
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    print("[SUCCESS] Picamera2 started!")

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [200, 300, 440, 350]  # Ground indicator line ROI

    # Navigation flags & state counters
    lTurn = False
    rTurn = False
    t = 0  # Completed turn count

    # Control parameters
    kp = 0.02
    kd = 0.006

    straightConst = 100  # ESP32 servo center (100 deg)
    tDeviation = 20

    # Turn angle thresholds (ESP32 servo range: 40 to 160 deg)
    sharpRight = straightConst + tDeviation  # 120 deg
    sharpLeft = straightConst - tDeviation   # 80 deg
    maxRight = 140
    maxLeft = 60

    turnThresh = 150   # Area below which turn begins
    exitThresh = 1500  # Area above which turn ends

    angle = straightConst
    prevAngle = angle
    aDiff = 0
    prevDiff = 0

    lDetected = False
    debug = "Debug" in sys.argv or "-d" in sys.argv
    start = False
    turnDir = "none"

    # Start robot driving forward
    serial_ctrl.send_command("FORWARD")
    time.sleep(0.5)

    try:
        while True:
            # Capture frame from Pi Camera v2
            img = picam2.capture_array()

            # Convert BGR to LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img_lab = cv2.GaussianBlur(img_lab, (7, 7), 0)

            # Find contours for left/right walls & orange/blue floor lines
            cListLeft = find_contours(img_lab, rBlack, ROI1)
            cListRight = find_contours(img_lab, rBlack, ROI2)
            cListOrange = find_contours(img_lab, rOrange, ROI3)
            cListBlue = find_contours(img_lab, rBlue, ROI3)

            # Calculate wall contour areas
            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]

            # Detect orange/blue corner marker lines
            if max_contour(cListOrange, ROI3)[0] > 100:
                lDetected = True
                if turnDir == "none":
                    turnDir = "right"
            elif max_contour(cListBlue, ROI3)[0] > 100:
                lDetected = True
                if turnDir == "none":
                    turnDir = "left"

            # Calculate area difference for wall centering
            aDiff = rightArea - leftArea

            # Compute steering angle using PD control algorithm
            # (Note: Positive aDiff means right wall is larger, so steer left -> decrease angle)
            angle = int(straightConst - (aDiff * kp + (aDiff - prevDiff) * kd))
            angle = max(maxLeft, min(maxRight, angle))

            # Turn triggering logic
            if leftArea <= turnThresh and not rTurn:
                lTurn = True
            elif rightArea <= turnThresh and not lTurn:
                rTurn = True

            # In a turn state
            if lTurn or rTurn:
                # Check turn completion condition
                if (rightArea > exitThresh and rTurn) or (leftArea > exitThresh and lTurn):
                    lTurn = False
                    rTurn = False
                    prevDiff = 0
                    if lDetected:
                        t += 1
                        print(f"[NAV EVENT] Completed turn {t}/12")
                        lDetected = False
                elif lTurn:
                    angle = max(angle, sharpLeft)
                elif rTurn:
                    angle = min(angle, sharpRight)
            else:
                # Clamp straight driving angle
                angle = max(sharpLeft, min(sharpRight, angle))

            # Transmit continuous steering angle to ESP32 (refreshes 500ms watchdog)
            serial_ctrl.send_steer(angle)

            # Update tracking variables
            prevDiff = aDiff
            prevAngle = angle

            # Stop after 3 full laps (12 turns)
            if t >= 12 and abs(angle - straightConst) <= 10:
                print(f"[FINISH] Completed 12 turns (3 laps). Stopping bot!")
                time.sleep(1.0 if turnDir == "left" else 1.5)
                serial_ctrl.send_command("STOP")
                break

            # Debug display
            if debug:
                img_disp = display_roi(img, [ROI1, ROI2, ROI3])
                cv2.drawContours(img_disp[ROI3[1]:ROI3[3], ROI3[0]:ROI3[2]], cListOrange, -1, (0, 255, 0), 2)
                cv2.drawContours(img_disp[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]], cListLeft, -1, (0, 255, 0), 2)
                cv2.drawContours(img_disp[ROI2[1]:ROI2[3], ROI2[0]:ROI2[2]], cListRight, -1, (0, 255, 0), 2)
                cv2.imshow("WRO Open Challenge (Pi 5)", img_disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[USER INTERRUPT] Stopping bot...")
                    serial_ctrl.send_command("STOP")
                    break

                display_variables({
                    "Left Area": leftArea,
                    "Right Area": rightArea,
                    "Steer Angle": angle,
                    "Turns": t,
                    "Marker Detected": lDetected
                })

            time.sleep(0.02)  # 50 Hz vision loop

    except KeyboardInterrupt:
        print("\n[SAFETY] Keyboard Interrupt. Halting bot...")
    finally:
        serial_ctrl.send_command("STOP")
        time.sleep(0.1)
        serial_ctrl.disconnect()
        if debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
