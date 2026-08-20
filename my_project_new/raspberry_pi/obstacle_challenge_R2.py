#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Obstacle Challenge Autonomous Navigation (Round 2)

Universal Camera Support:
- Supports both Pi Camera v2 (Picamera2) AND standard USB Webcams (cv2.VideoCapture).
- Auto-fallbacks to USB Webcam if Picamera2 is unavailable.
- Can be forced to use USB Webcam with '--webcam' or '-w' flag.
"""

import sys
import time
import math
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rMagenta, rRed, rGreen, rBlue, rOrange, rBlack, lotType
from wro_functions import find_contours, max_contour, display_roi, display_variables
from camera_streamer import CameraDebugStreamer
from open_challenge_R1 import CameraManager


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
    print("   Universal Camera Support (Picamera2 & USB Webcam)")
    print("=" * 65)

    force_webcam = "--webcam" in sys.argv or "-w" in sys.argv

    # 1. Initialize USB Serial connection to ESP32
    serial_ctrl = WROSerialController()
    if not serial_ctrl.connect():
        print("[ERROR] Cannot proceed without ESP32 serial connection.")
        sys.exit(1)

    # Force immediate STOP during camera warmup
    print("[SAFETY] Forcing robot STOP state during initialization...")
    serial_ctrl.send_command("STOP")
    time.sleep(0.5)

    # 2. Start Live Web Camera Debug Streamer (port 8080)
    streamer = CameraDebugStreamer(port=8080)
    streamer.start()

    show_monitor_display = "--no-display" not in sys.argv
    window_name = "WRO Obstacle Challenge - Monitor Debug View (Pi 5)"

    if show_monitor_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            print("[DISPLAY] Created OpenCV live display window on monitor!")
        except Exception as e:
            print(f"[WARNING] Could not open GUI display window: {e}")
            show_monitor_display = False

    # 3. Initialize Camera (Picamera2 or USB Webcam)
    camera = CameraManager(force_webcam=force_webcam, device_index=0)
    camera.start()

    # Warmup camera frames
    print("[INFO] Capturing camera warmup frames...")
    for _ in range(15):
        warmup_frame = camera.capture_array()
        if warmup_frame is not None:
            streamer.update_frame(warmup_frame)
            if show_monitor_display:
                cv2.imshow(window_name, warmup_frame)
                cv2.waitKey(1)
        time.sleep(0.04)

    print("\n[READY] Camera & Vision Engine Ready!")
    print("[COUNTDOWN] Bot starts driving in 3 seconds... (Press 'q' to abort)")
    for c in range(3, 0, -1):
        print(f"[COUNTDOWN] {c}...")
        if show_monitor_display and warmup_frame is not None:
            cd_img = warmup_frame.copy()
            cv2.putText(cd_img, f"STARTING IN {c} SECONDS...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow(window_name, cd_img)
            cv2.waitKey(1)
        time.sleep(1.0)

    print("[START] Driving FORWARD now!")
    serial_ctrl.send_command("FORWARD")

    # Regions of Interest (ROI)
    ROI1 = [10, 150, 260, 240]   # Left wall ROI
    ROI2 = [380, 150, 630, 240]  # Right wall ROI
    ROI3 = [200, 240, 440, 360]  # Ground / Pillar ROI

    redTarget = 120    # Target X coordinate when keeping Red pillar on left
    greenTarget = 520  # Target X coordinate when keeping Green pillar on right
    straightConst = 100
    targetWallArea = 2200
    wallMinArea = 200

    t = 0  # Completed lap/turn counter
    navMode = "TRACKING"

    try:
        while True:
            img = camera.capture_array()
            if img is None:
                time.sleep(0.01)
                continue

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

            if p_red.area > 0 and p_red.dist < p_green.dist:
                navMode = "RED_PILLAR"
                error = p_red.x - redTarget
                steer_angle = int(straightConst - (error * 0.15))
            elif p_green.area > 0:
                navMode = "GREEN_PILLAR"
                error = p_green.x - greenTarget
                steer_angle = int(straightConst - (error * 0.15))
            elif leftArea > wallMinArea and rightArea > wallMinArea:
                navMode = "DUAL_WALL"
                aDiff = rightArea - leftArea
                steer_angle = int(straightConst - (aDiff * 0.02))
            elif leftArea > wallMinArea:
                navMode = "SINGLE_LEFT"
                wallError = leftArea - targetWallArea
                steer_angle = int(straightConst + (wallError * 0.015))
            elif rightArea > wallMinArea:
                navMode = "SINGLE_RIGHT"
                wallError = rightArea - targetWallArea
                steer_angle = int(straightConst - (wallError * 0.015))
            else:
                navMode = "SEARCHING"
                steer_angle = straightConst

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

            img_disp = img.copy()
            img_disp = display_roi(img_disp, [ROI1, ROI2, ROI3])
            cam_type = "WEBCAM" if camera.is_webcam else "PICAM2"
            telemetry_text = f"Cam: {cam_type} | Mode: {navMode} | Steer: {steer_angle} | RedDist: {p_red.dist}"
            cv2.putText(img_disp, telemetry_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 204), 2)

            streamer.update_frame(img_disp)

            if show_monitor_display:
                cv2.imshow(window_name, img_disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[USER INTERRUPT] Stopping bot from monitor GUI...")
                    serial_ctrl.send_command("STOP")
                    break

            display_variables({
                "Camera Type": cam_type,
                "Nav Mode": navMode,
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
        streamer.stop()
        camera.stop()
        time.sleep(0.1)
        serial_ctrl.disconnect()
        if show_monitor_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
