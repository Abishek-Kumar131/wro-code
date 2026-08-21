#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Obstacle Challenge Autonomous Navigation (Round 2)

Universal Camera & Sensor Support:
- Uses Dual-Layer HSV+LAB Black Wall Segmentation with explicit HSV Blue/Orange Mask Subtraction (0% Overlap).
- Red Pillar Avoidance: Passes Red pillars on the RIGHT (keeps pillar on left side of car).
- Green Pillar Avoidance: Passes Green pillars on the LEFT (keeps pillar on right side of car).
- Camera Vision Wall Avoidance: Dynamically steers AWAY from black side walls when no pillars are nearby.
- Parking Lot Head-In Parking: Detects Magenta parking zone at course completion.
"""

import sys
import time
import math
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rMagenta, rRed, rGreen, rBlue, rOrange, rBlack, lotType
from wro_functions import (CameraManager, find_black_wall_contours, find_contours, max_contour, draw_roi,
                           draw_offset_contours, display_variables)
from camera_streamer import CameraDebugStreamer


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
    print("   Hybrid Architecture: Pillar Avoidance + Vision Wall Centering")
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

    print("[START] Driving FORWARD with Pillar Avoidance!")
    serial_ctrl.send_command("FORWARD")

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [180, 220, 460, 360]  # Ground / Pillar ROI

    redTarget = 120    # Target X coordinate when keeping Red pillar on left
    greenTarget = 520  # Target X coordinate when keeping Green pillar on right
    straightConst = 100
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

            # Extract contours using dual-layer HSV+LAB black wall segmentation (100% blue exclusion)
            cListLeft = find_black_wall_contours(img, ROI1)
            cListRight = find_black_wall_contours(img, ROI2)
            cListRed = find_contours(img_lab, rRed, ROI3)
            cListGreen = find_contours(img_lab, rGreen, ROI3)
            cListMagenta = find_contours(img_lab, rMagenta, ROI3)

            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]
            magentaArea = max_contour(cListMagenta, ROI3)[0]

            # Get latest ultrasonic sensor telemetry from ESP32
            us_data = serial_ctrl.get_us_data()
            f_us = us_data.get("f", 0)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

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
            elif leftArea > wallMinArea or rightArea > wallMinArea:
                navMode = "VISION_WALLS"
                aDiff = rightArea - leftArea  # Negative when close to left wall
                steer_angle = int(straightConst - (aDiff * 0.02))
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
                serial_ctrl.send_command("STEER:100")
                time.sleep(1.0)
                serial_ctrl.send_command("STOP")
                print("[FINISH] Parking complete!")
                break

            # Stream drive command over USB serial (motor speed 245 + dynamic steer angle)
            serial_ctrl.send_command("AUTO_US_OFF")
            serial_ctrl.send_command(f"DRIVE:245:{steer_angle}")

            # Draw ROIs & Offset Contours (matching open_challenge_R1.py)
            img_disp = img.copy()
            draw_roi(img_disp, ROI1, (0, 255, 255), 2)
            draw_roi(img_disp, ROI2, (0, 255, 255), 2)
            draw_roi(img_disp, ROI3, (255, 255, 0), 2)
            draw_offset_contours(img_disp, cListLeft, ROI1, (0, 255, 0), 2)
            draw_offset_contours(img_disp, cListRight, ROI2, (0, 255, 0), 2)
            draw_offset_contours(img_disp, cListRed, ROI3, (0, 0, 255), 2)      # Red contour for Red Pillar
            draw_offset_contours(img_disp, cListGreen, ROI3, (0, 255, 0), 2)    # Green contour for Green Pillar
            draw_offset_contours(img_disp, cListMagenta, ROI3, (255, 0, 255), 2)# Magenta contour for Parking Lot

            cam_type = "WEBCAM" if camera.is_webcam else "PICAM2"
            telemetry_text = f"Cam:{cam_type} | Mode:{navMode} | Steer:{steer_angle} | RedDist:{p_red.dist}"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | Mag:{magentaArea}px"
            us_text = f"US Sensors -> F:{f_us}cm | L:{l_us}cm | R:{r_us}cm | B:{b_us}cm"

            cv2.putText(img_disp, telemetry_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 204), 2)
            cv2.putText(img_disp, wall_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(img_disp, us_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
                "Steer Angle": steer_angle,
                "Red Pillar Dist": p_red.dist,
                "Green Pillar Dist": p_green.dist,
                "Left Wall Area (px)": leftArea,
                "Right Wall Area (px)": rightArea,
                "Magenta Area (px)": magentaArea,
                "US Front (cm)": f_us,
                "US Left (cm)": l_us,
                "US Right (cm)": r_us,
                "US Back (cm)": b_us
            })

            time.sleep(0.02)  # 50 Hz vision loop

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
