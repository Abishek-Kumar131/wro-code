#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Obstacle Challenge Autonomous Navigation (Round 2)

Architecture:
- Raspberry Pi 5 runs OpenCV vision with Picamera2 (Pi Cam v2)
- Pillar Object Tracking (Red/Green) & Magenta Parking Space Detection
- Streams commands over USB Serial to ESP32 controller (115200 baud)
- ESP32 controls DC motor & steering servo with 500ms watchdog failsafe
"""

import sys
import time
import math
import cv2
import numpy as np
from picamera2 import Picamera2
from wro_serial import WROSerialController
from masks import rMagenta, rRed, rGreen, rBlue, rOrange, rBlack, lotType
from wro_functions import find_contours, max_contour, display_roi, display_variables


class Pillar:
    def __init__(self, area, dist, x, y, target):
        self.area = area
        self.dist = dist
        self.x = x
        self.y = y
        self.target = target
        self.w = 0
        self.h = 0

    def set_dimensions(self, w, h):
        self.w = w
        self.h = h


def find_pillar(contours, target, p, colour, ROI3, tempParking=False):
    """Processes pillar contours and returns the nearest pillar candidate."""
    num_p = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if (area > 150 and colour == "red") or (area > 100 and colour == "red" and tempParking) or (area > 200 and colour == "green"):
            if tempParking and colour == "green" and area < 300:
                continue

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            x += ROI3[0] + w // 2
            y += ROI3[1] + h

            temp_dist = round(math.dist([x, y], [320, 480]), 0)

            if 160 < temp_dist < 380:
                num_p += 1

            if temp_dist < p.dist:
                p.area = area
                p.dist = temp_dist
                p.y = y
                p.x = x
                p.target = target
                p.set_dimensions(w, h)

    return p, num_p


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 2 Obstacle Challenge Node (Pi 5)")
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

    # Regions of Interest (ROI)
    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [200, 240, 440, 360]  # Ground / Pillar ROI

    redTarget = 120    # Target X coordinate when keeping Red pillar on left
    greenTarget = 520  # Target X coordinate when keeping Green pillar on right
    straightConst = 100

    debug = "Debug" in sys.argv or "-d" in sys.argv
    t = 0  # Completed lap/turn counter

    serial_ctrl.send_command("FORWARD")
    time.sleep(0.5)

    try:
        while True:
            img = picam2.capture_array()
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img_lab = cv2.GaussianBlur(img_lab, (7, 7), 0)

            cListLeft = find_contours(img_lab, rBlack, ROI1)
            cListRight = find_contours(img_lab, rBlack, ROI2)
            cListRed = find_contours(img_lab, rRed, ROI3)
            cListGreen = find_contours(img_lab, rGreen, ROI3)
            cListMagenta = find_contours(img_lab, rMagenta, ROI3)

            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]
            magentaArea = max_contour(cListMagenta, ROI3)[0]

            # Detect nearest Red and Green pillars
            p_red = Pillar(0, 999, 0, 0, redTarget)
            p_red, num_red = find_pillar(cListRed, redTarget, p_red, "red", ROI3)

            p_green = Pillar(0, 999, 0, 0, greenTarget)
            p_green, num_green = find_pillar(cListGreen, greenTarget, p_green, "green", ROI3)

            # Determine active steering target
            steer_angle = straightConst

            # If Red pillar is close, steer to pass Red on right (target redTarget ~ 120)
            if p_red.area > 0 and p_red.dist < p_green.dist:
                error = p_red.x - redTarget
                steer_angle = int(straightConst - (error * 0.15))
            # If Green pillar is close, steer to pass Green on left (target greenTarget ~ 520)
            elif p_green.area > 0:
                error = p_green.x - greenTarget
                steer_angle = int(straightConst - (error * 0.15))
            else:
                # Default wall-centering PD calculation
                aDiff = rightArea - leftArea
                steer_angle = int(straightConst - (aDiff * 0.02))

            steer_angle = max(60, min(140, steer_angle))

            # Emergency reversing if blocked directly by pillar
            if (p_red.area > 6500 and p_red.y > 350) or (p_green.area > 8000 and p_green.y > 350):
                print("[SAFETY] Blocked by pillar! Reversing...")
                serial_ctrl.send_command("BACKWARD")
                time.sleep(0.6)
                serial_ctrl.send_command("FORWARD")
                continue

            # Parking lot detection & head-in park
            if magentaArea > 3500 and t >= 12:
                print("[PARKING] Magenta parking lot detected! Head-in parking...")
                serial_ctrl.send_steer(straightConst)
                time.sleep(1.0)
                serial_ctrl.send_command("STOP")
                print("[FINISH] Parking complete!")
                break

            # Send continuous steering angle over USB Serial (refreshes 500ms watchdog)
            serial_ctrl.send_steer(steer_angle)

            if debug:
                img_disp = display_roi(img, [ROI1, ROI2, ROI3])
                cv2.imshow("WRO Obstacle Challenge (Pi 5)", img_disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    serial_ctrl.send_command("STOP")
                    break

                display_variables({
                    "Red Dist": p_red.dist,
                    "Green Dist": p_green.dist,
                    "Steer Angle": steer_angle,
                    "Magenta Area": magentaArea
                })

            time.sleep(0.02)

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
