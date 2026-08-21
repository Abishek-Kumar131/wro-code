#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Obstacle Challenge Autonomous Navigation (Round 2)

Integrated ObstacleChallengeV2 Architecture:
- PD Steering for Pillar Avoidance with Vertical Y Proximity Scaling (cKp=0.25, cKd=0.25, cy=0.08).
- PD Steering for Wall Centering (kp=0.015, kd=0.01).
- Dual-Layer HSV+LAB Black Wall Segmentation with 100% Pillar & Line Color Exclusion.
- Emergency Pillar Reversing & Safety Collision Prevention.
- Advanced Lap 3 Magenta Parking Lot Navigation (Left/Right Lot Head-In Parking).
- Adapted for Raspberry Pi 5 + ESP32 USB Serial Controller.
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


class Pillar:
    def __init__(self, area, dist, x, y, target):
        self.area = area       # Pillar area
        self.dist = dist       # Distance from bottom-middle of screen (320, 480)
        self.x = x             # Pillar X coordinate
        self.y = y             # Pillar Y coordinate
        self.target = target   # Target X position (redTarget or greenTarget)
        self.w = 0
        self.h = 0

    def set_dimensions(self, w, h):
        self.w = w
        self.h = h


def find_pillar(contours, target, p, colour, ROI3, tempParking=False, leftArea=0, rightArea=0, maxDist=370, endConst=30):
    """Processes pillar contours and returns the nearest pillar candidate matching ObstacleChallengeV2 logic."""
    num_p = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Check if area is large enough for the specific color pillar
        if (area > 150 and colour == "red") or (area > 100 and colour == "red" and tempParking) or (area > 200 and colour == "green"):
            if tempParking and colour == "green" and area < 300:
                continue

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            # Convert ROI-relative coordinates to full-frame coordinates
            x += ROI3[0] + w // 2
            y += ROI3[1] + h

            # Distance between pillar bottom and screen bottom-center (320, 480)
            temp_dist = round(math.dist([x, y], [320, 480]), 0)

            if 160 < temp_dist < 380:
                num_p += 1

            # Skip pillar if it gets too close to bottom ROI or exceeds max distance
            if y > ROI3[3] - endConst or temp_dist > maxDist:
                continue

            # Update if this pillar is closer than previous candidate
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
    print("   Architecture: Integrated ObstacleChallengeV2 PD Control Engine")
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

    show_monitor_display = "--no-display" not in sys.argv
    window_name = "WRO Obstacle Challenge - V2 Debug View (Pi 5)"

    if show_monitor_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            print("[DISPLAY] Created OpenCV live display window on monitor!")
        except Exception as e:
            print(f"[WARNING] Could not open GUI display window: {e}")
            show_monitor_display = False

    # 2. Initialize Camera (Picamera2 or USB Webcam)
    camera = CameraManager(force_webcam=force_webcam, device_index=0)
    camera.start()

    # Warmup camera frames
    print("[INFO] Capturing camera warmup frames...")
    for _ in range(15):
        warmup_frame = camera.capture_array()
        if warmup_frame is not None:
            if show_monitor_display:
                cv2.imshow(window_name, warmup_frame)
                cv2.waitKey(1)
        time.sleep(0.04)

    # 3. Safety Countdown before bot starts driving
    print("\n[READY] Obstacle V2 Control Engine Ready!")
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

    print("[START] Driving FORWARD with ObstacleV2 PD Steering Engine!")
    serial_ctrl.send_command("FORWARD")

    # ------------------------------------------------------------------------
    # Initialization of ObstacleChallengeV2 Parameters
    # ------------------------------------------------------------------------
    redTarget = 110    # Target X position for Red Pillars (Keep on LEFT)
    greenTarget = 530  # Target X position for Green Pillars (Keep on RIGHT)

    straightConst = 100 # Steering center (100 degrees)
    sharpRight = 60    # Sharp right steering lock (100 - 40)
    sharpLeft = 140    # Sharp left steering lock (100 + 40)
    motorSpeed = 245   # PWM speed

    # PD Wall-Centering gains
    kp = 0.015
    kd = 0.01

    # PD Pillar Avoidance gains
    cKp = 0.25
    cKd = 0.25
    cy = 0.08

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    ROI1 = [0, 175, 330, 265]   # Left Wall ROI
    ROI2 = [330, 175, 640, 265]  # Right Wall ROI
    ROI3 = [redTarget - 50, 120, greenTarget + 50, 345] # Signal Pillars ROI
    ROI4 = [200, 260, 440, 310]  # Ground Markers & Parking Lot ROI

    # Navigation state variables
    turnDir = "none"
    t = 0
    t2 = 0
    lTurn = False
    rTurn = False
    tempParking = False
    parkingL = False
    parkingR = False

    angle = straightConst
    prevAngle = angle
    aDiff = 0
    prevDiff = 0
    error = 0
    prevError = 0
    endConst = 30
    maxDist = 370

    try:
        while True:
            img = camera.capture_array()
            if img is None:
                time.sleep(0.01)
                continue

            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

            # Extract contours using dual-layer HSV+LAB black wall segmentation (100% color exclusion)
            contours_left = find_black_wall_contours(img, ROI1)
            contours_right = find_black_wall_contours(img, ROI2)
            contours_red = find_contours(img_lab, rRed, ROI3)
            contours_green = find_contours(img_lab, rGreen, ROI3)
            contours_orange = find_contours(img_lab, rOrange, ROI4)
            contours_blue = find_contours(img_lab, rBlue, ROI4)
            contours_magenta = find_contours(img_lab, rMagenta, ROI4)

            leftArea = max_contour(contours_left, ROI1)[0]
            rightArea = max_contour(contours_right, ROI2)[0]
            maxO = max_contour(contours_orange, ROI4)[0]
            maxB = max_contour(contours_blue, ROI4)[0]
            magentaArea = max_contour(contours_magenta, ROI4)[0]

            # Get latest ultrasonic sensor telemetry from ESP32
            us_data = serial_ctrl.get_us_data()
            f_us = us_data.get("f", 0)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

            # -------------------------------------------------------------
            # Nearest Pillar Tracking (ObstacleChallengeV2 Logic)
            # -------------------------------------------------------------
            temp_p = Pillar(0, 1000000, 0, 0, greenTarget)
            cPillar, num_pillars_g = find_pillar(contours_green, greenTarget, temp_p, "green", ROI3, tempParking, leftArea, rightArea, maxDist, endConst)
            cPillar, num_pillars_r = find_pillar(contours_red, redTarget, cPillar, "red", ROI3, tempParking, leftArea, rightArea, maxDist, endConst)

            # Dynamically adjust PD gains based on pillar density
            if num_pillars_g >= 2 or num_pillars_r >= 2:
                endConst = 60
                cKp, cKd, cy = 0.20, 0.20, 0.05
            else:
                endConst = 30
                cKp, cKd, cy = 0.25, 0.25, 0.08

            # -------------------------------------------------------------
            # Track Turn Direction Detection (Orange = Right, Blue = Left)
            # -------------------------------------------------------------
            if turnDir == "none":
                if maxO > 100 and maxO > maxB:
                    turnDir = "right"
                    print("[VISION MARKER] Detected ORANGE Line -> Track Direction = RIGHT")
                elif maxB > 100 and maxB > maxO:
                    turnDir = "left"
                    print("[VISION MARKER] Detected BLUE Line -> Track Direction = LEFT")

            if (turnDir == "right" and maxO > 100) or (turnDir == "left" and maxB > 100):
                if turnDir == "right":
                    rTurn = True
                else:
                    lTurn = True

            # -------------------------------------------------------------
            # Servo Steering Calculations (ObstacleChallengeV2 PD Engine)
            # -------------------------------------------------------------
            navMode = "SEARCHING"

            # Case A: No Pillar Detected -> PD Wall-Centering
            if cPillar.area == 0 and not parkingL and not parkingR:
                navMode = "VISION_WALLS"
                aDiff = rightArea - leftArea  # Negative when close to left wall
                angle = int(straightConst - (aDiff * kp) - ((aDiff - prevDiff) * kd))
                prevDiff = aDiff

            # Case B: Pillar Detected -> PD Pillar Avoidance + Y Proximity Scaling
            elif not parkingR and not parkingL:
                navMode = "RED_PILLAR" if cPillar.target == redTarget else "GREEN_PILLAR"
                
                # Calculate X error relative to target position
                error = cPillar.target - cPillar.x
                angle = int(straightConst - (error * cKp) - ((error - prevError) * cKd))

                # Adjust angle further based on vertical proximity (cy scaling)
                if not tempParking:
                    y_offset = int(cy * (cPillar.y - ROI3[1]))
                    angle -= y_offset if error <= 0 else -y_offset

                prevError = error

            # -------------------------------------------------------------
            # Emergency Reversing Safety Check (Blocked directly by pillar)
            # -------------------------------------------------------------
            if ((cPillar.area > 6500 and cPillar.target == redTarget) or 
                (cPillar.area > 8000 and cPillar.target == greenTarget)) and cPillar.y > 350 and not tempParking:
                print("[SAFETY] Dangerously close to pillar! Executing emergency reverse...")
                serial_ctrl.send_command("BACKWARD")
                time.sleep(0.6)
                serial_ctrl.send_command("FORWARD")
                continue

            # -------------------------------------------------------------
            # Final Lap Magenta Parking Lot Algorithm (Lap 3: t >= 12)
            # -------------------------------------------------------------
            if t >= 12 and not tempParking and ((leftArea > 2000 and rightArea > 2000 and cPillar.area < 1000) or cPillar.area < 400):
                print("[PARKING] Searching for Magenta Parking Lot...")
                tempParking = True

            if tempParking:
                if magentaArea > 3000:
                    navMode = "PARKING_LOT"
                    print("[PARKING] Entering Magenta Parking Lot!")
                    # Head-in park into parking space
                    angle = sharpLeft if turnDir == "left" else sharpRight
                    serial_ctrl.send_command(f"DRIVE:{motorSpeed}:{angle}")
                    time.sleep(1.2)
                    serial_ctrl.send_command("STOP")
                    print("[FINISH] Obstacle Challenge Complete!")
                    break

            # Constrain angle between safe mechanical limits (60 to 140 deg)
            angle = max(60, min(140, angle))

            # Stream continuous DRIVE command over USB Serial to ESP32
            serial_ctrl.send_command("AUTO_US_OFF")
            serial_ctrl.send_command(f"DRIVE:{motorSpeed}:{angle}")

            # Draw ROIs & Offset Contours (matching open_challenge_R1.py)
            img_disp = img.copy()
            draw_roi(img_disp, ROI1, (0, 255, 255), 2)
            draw_roi(img_disp, ROI2, (0, 255, 255), 2)
            draw_roi(img_disp, ROI3, (255, 255, 0), 2)
            draw_roi(img_disp, ROI4, (255, 0, 255), 2)
            draw_offset_contours(img_disp, contours_left, ROI1, (0, 255, 0), 2)
            draw_offset_contours(img_disp, contours_right, ROI2, (0, 255, 0), 2)
            draw_offset_contours(img_disp, contours_red, ROI3, (0, 0, 255), 2)       # Red contour for Red Pillar
            draw_offset_contours(img_disp, contours_green, ROI3, (0, 255, 0), 2)     # Green contour for Green Pillar
            draw_offset_contours(img_disp, contours_magenta, ROI4, (255, 0, 255), 2) # Magenta contour for Parking Lot

            cam_type = "WEBCAM" if camera.is_webcam else "PICAM2"
            telemetry_text = f"Cam:{cam_type} | Mode:{navMode} | Steer:{angle} | PillarDist:{cPillar.dist}"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | Mag:{magentaArea}px"
            us_text = f"US Sensors -> F:{f_us}cm | L:{l_us}cm | R:{r_us}cm | B:{b_us}cm"

            cv2.putText(img_disp, telemetry_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 204), 2)
            cv2.putText(img_disp, wall_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(img_disp, us_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
                "Steer Angle": angle,
                "Pillar Dist": cPillar.dist,
                "Pillar X": cPillar.x,
                "Left Wall Area (px)": leftArea,
                "Right Wall Area (px)": rightArea,
                "Magenta Area (px)": magentaArea,
                "US Front (cm)": f_us,
                "US Left (cm)": l_us,
                "US Right (cm)": r_us,
                "US Back (cm)": b_us
            })

            prevAngle = angle
            time.sleep(0.02)  # 50 Hz vision loop

    except KeyboardInterrupt:
        print("\n[SAFETY] Keyboard Interrupt. Halting bot...")
    finally:
        serial_ctrl.send_command("STOP")
        camera.stop()
        time.sleep(0.1)
        serial_ctrl.disconnect()
        if show_monitor_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
