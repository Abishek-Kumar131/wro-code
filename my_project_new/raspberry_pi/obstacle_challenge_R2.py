#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Obstacle Challenge Autonomous Navigation (Round 2)

Hybrid Vision Architecture:
- Anti-Drift Dynamic Speed Control: Automatically reduces motor speed from 245 to 195
  when steering angle deflection exceeds 30 degrees (angle < 70 or angle > 130) or during corner turns.
- Permanent First-Color Direction Lock: Locks onto whichever marker color (Orange/Blue) is detected first.
- Exact Corner Turn Logic & Vision-Dynamic Turn Exit from open_challenge_R1.py:
  * Triggers turn on Floor Marker Line + Corner Wall Drop.
  * Vision-Dynamic Turn Exit: Dynamically exits corner turn as soon as the camera re-acquires the new straightaway wall (min 0.8s, max 2.2s).
  * Decoupled 3.5s Line Lockout and Turn Cooldown timers.
- Integrated ObstacleChallengeV2 Straightaway Steering Engine:
  * PD Steering for Red/Green Pillar Avoidance with Vertical Y Proximity Scaling (cKp=0.25, cKd=0.25, cy=0.08).
  * PD Steering for Wall Centering (kp=0.015, kd=0.01).
- Triple Safeguard Red/Orange Separation (0% Red/Orange Overlap).
- Emergency Pillar Reversing & Safety Collision Prevention.
- Advanced Lap 3 Magenta Parking Lot Navigation (Head-In Parking at t >= 12).
"""

import sys
import time
import math
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rMagenta, rRed, rGreen, rBlue, rOrange, rBlack, lotType
from wro_functions import (CameraManager, find_black_wall_contours, find_red_pillar_contours,
                           find_orange_line_contours, find_contours, max_contour, draw_roi,
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


def find_pillar(contours, target, p, colour, ROI3, tempParking=False, maxDist=370, endConst=30):
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
    print("   Architecture: Anti-Drift Dynamic Speed Control + Dynamic Vision Turn Exit")
    print("=" * 65)

    force_webcam = "--webcam" in sys.argv or "-w" in sys.argv

    # Check for direction override in arguments (--dir left or --dir right)
    forced_dir = "none"
    if "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 < len(sys.argv):
            forced_dir = sys.argv[idx + 1].lower()
            print(f"[CONFIG] Forcing fixed track direction: {forced_dir.upper()}")

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
    window_name = "WRO Obstacle Challenge - Round 2 Monitor Debug (Pi 5)"

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
    print("\n[READY] Obstacle Round 2 Engine Ready!")
    print("[COUNTDOWN] Bot starts driving in 3 seconds... (Press 'q' to abort, 'l'/'r' to set dir)")
    for c in range(3, 0, -1):
        print(f"[COUNTDOWN] {c}...")
        if show_monitor_display and warmup_frame is not None:
            cd_img = warmup_frame.copy()
            cv2.putText(cd_img, f"STARTING IN {c} SECONDS...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow(window_name, cd_img)
            cv2.waitKey(1)
        time.sleep(1.0)

    print("[START] Driving FORWARD with Anti-Drift Speed Control & Dynamic Vision Turn Exit!")
    serial_ctrl.send_command("FORWARD")

    # ------------------------------------------------------------------------
    # Obstacle & Steering Parameters
    # ------------------------------------------------------------------------
    redTarget = 110    # Target X position for Red Pillars (Keep on LEFT)
    greenTarget = 530  # Target X position for Green Pillars (Keep on RIGHT)

    straightConst = 100 # Steering center (100 degrees)
    sharpRight = 60    # Sharp right steering lock
    sharpLeft = 140    # Sharp left steering lock

    # Speed Parameters (Anti-Drift Speed Control)
    normalSpeed = 245  # Full straightaway speed (96% PWM)
    turnSpeed = 195    # Reduced cornering speed when steering deflection > 30° to prevent drifting!

    # PD Wall-Centering gains
    kp = 0.015
    kd = 0.01

    # PD Pillar Avoidance gains
    cKp = 0.25
    cKd = 0.25
    cy = 0.08

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    # Vertically separated: ROI3 (Pillars) Y <= 245 vs ROI4 (Floor Lines) Y >= 260!
    ROI1 = [0, 175, 330, 265]   # Left Wall ROI
    ROI2 = [330, 175, 640, 265]  # Right Wall ROI
    ROI3 = [redTarget - 50, 110, greenTarget + 50, 245] # Signal Pillars ROI (Standing 3D Pillars above horizon!)
    ROI4 = [200, 260, 440, 330]  # Ground Markers & Parking Lot ROI (Floor Lines)

    # Navigation flags & state counters
    t = 0                  # Completed turn count (3 laps x 4 turns = 12)
    turnDir = forced_dir   # Track direction ("left", "right", or "none")
    lDetected = False
    isTurning = False
    turnStartTime = 0
    lineLockoutUntil = 0   # 3.5s line detection lockout timer
    turnCooldownUntil = 0  # 3.5s turn trigger cooldown timer
    lockoutDuration = 3.5  # Exactly 3.5 seconds lockout

    # Dynamic Turn Exit Timings (Optimized for Narrow FOV Camera)
    minTurnDuration = 0.8  # Minimum arc turn time before checking wall re-acquisition (0.8s)
    maxTurnDuration = 2.2  # Safety maximum turn time cap (2.2s)
    wallReacquireArea = 600 # Area threshold to confirm single wall in narrow FOV view
    turnThresh = 200       # Area threshold below which wall end is detected

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

    # Rate-limiting variables
    last_steer_angle = None
    last_drive_speed = None
    last_cmd_time = 0

    try:
        while True:
            img = camera.capture_array()
            if img is None:
                time.sleep(0.01)
                continue

            currTime = time.time()
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

            # Extract contours using strict separation logic
            contours_left = find_black_wall_contours(img, ROI1)
            contours_right = find_black_wall_contours(img, ROI2)
            contours_red = find_red_pillar_contours(img, ROI3)       # Strict Red Pillar + Aspect Ratio filter (H/W >= 0.75)
            contours_green = find_contours(img_lab, rGreen, ROI3)
            contours_orange = find_orange_line_contours(img, ROI4)    # Strict Orange Line (Floor ROI4: Y in 260..330)
            contours_blue = find_contours(img_lab, rBlue, ROI4)
            contours_magenta = find_contours(img_lab, rMagenta, ROI4)

            leftArea = max_contour(contours_left, ROI1)[0]
            rightArea = max_contour(contours_right, ROI2)[0]
            orangeArea = max_contour(contours_orange, ROI4)[0]
            blueArea = max_contour(contours_blue, ROI4)[0]
            magentaArea = max_contour(contours_magenta, ROI4)[0]

            # Get latest ultrasonic sensor telemetry from ESP32
            us_data = serial_ctrl.get_us_data()
            f_us = us_data.get("f", 0)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

            # -------------------------------------------------------------
            # 1. PERMANENT FIRST-COLOR DIRECTION LOCK & MARKER DETECTION
            # -------------------------------------------------------------
            if not isTurning and currTime >= lineLockoutUntil:
                if turnDir == "none":
                    if orangeArea > 150 and orangeArea > blueArea:
                        turnDir = "right"
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[FIRST-COLOR LOCK] First Line Detected: ORANGE ({orangeArea} px) -> Permanently Locking Direction to RIGHT!")
                    elif blueArea > 150 and blueArea > orangeArea:
                        turnDir = "left"
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[FIRST-COLOR LOCK] First Line Detected: BLUE ({blueArea} px) -> Permanently Locking Direction to LEFT!")
                
                elif turnDir == "right":
                    if orangeArea > 150:
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[LOCKED MARKER] Detected ORANGE Line ({orangeArea} px) -> Track Dir = RIGHT (3.5s Line Lockout)")
                
                elif turnDir == "left":
                    if blueArea > 150:
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[LOCKED MARKER] Detected BLUE Line ({blueArea} px) -> Track Dir = LEFT (3.5s Line Lockout)")

            # -------------------------------------------------------------
            # 2. HYBRID CORNER TURN & DYNAMIC VISION EXIT (REDUCED TURN SPEED)
            # -------------------------------------------------------------
            if isTurning:
                targetTurnAngle = 140 if turnDir == "left" else 60
                
                # Stream active corner turn at reduced speed (195) to prevent drifting!
                if (currTime - last_cmd_time) >= 0.1 or last_drive_speed != turnSpeed:
                    serial_ctrl.send_command(f"DRIVE:{turnSpeed}:{targetTurnAngle}")
                    last_cmd_time = currTime
                    last_drive_speed = turnSpeed
                    last_steer_angle = targetTurnAngle

                turnElapsed = currTime - turnStartTime

                # DYNAMIC TURN EXIT CONDITION:
                newWallAcquired = (turnElapsed >= minTurnDuration) and (leftArea >= wallReacquireArea or rightArea >= wallReacquireArea)
                maxTimeoutReached = (turnElapsed >= maxTurnDuration)

                if newWallAcquired or maxTimeoutReached:
                    isTurning = False
                    turnCooldownUntil = currTime + lockoutDuration  # 3.5s cooldown after turn ends
                    lineLockoutUntil = currTime + lockoutDuration   # 3.5s line lockout after turn ends
                    exit_reason = "WALL_REACQUIRED" if newWallAcquired else "MAX_TIMEOUT"
                    print(f"[NAV EVENT] Turn {t}/12 ({turnDir.upper()}) EXITED via {exit_reason} in {round(turnElapsed, 2)}s!")
            
            elif currTime >= turnCooldownUntil:
                # Wall drop check (wall area drops below turnThresh)
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

                # STRICT TRIGGER: Require line marker detection (or forced dir) AND wall drop!
                if (lDetected or forced_dir != "none") and wallDropDetected:
                    targetTurnAngle = 140 if turnDir == "left" else 60
                    t += 1
                    print(f"[NAV EVENT] Marker Seen + Wall Drop! (L:{leftArea} R:{rightArea}) -> Triggering Turn ({t}/12) at speed {turnSpeed}...")
                    serial_ctrl.send_command(f"DRIVE:{turnSpeed}:{targetTurnAngle}")
                    last_cmd_time = currTime
                    last_drive_speed = turnSpeed
                    last_steer_angle = targetTurnAngle
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False  # Reset marker flag for next straightaway!
                    turnCooldownUntil = currTime + maxTurnDuration + lockoutDuration

            # -------------------------------------------------------------
            # 3. STRAIGHTAWAY OBSTACLE AVOIDANCE & DYNAMIC ANTI-DRIFT SPEED CONTROL
            # -------------------------------------------------------------
            if not isTurning:
                # Nearest Pillar Tracking (ObstacleChallengeV2 Logic)
                temp_p = Pillar(0, 1000000, 0, 0, greenTarget)
                cPillar, num_pillars_g = find_pillar(contours_green, greenTarget, temp_p, "green", ROI3, tempParking, maxDist, endConst)
                cPillar, num_pillars_r = find_pillar(contours_red, redTarget, cPillar, "red", ROI3, tempParking, maxDist, endConst)

                # Dynamically adjust PD gains based on pillar density
                if num_pillars_g >= 2 or num_pillars_r >= 2:
                    endConst = 60
                    cKp, cKd, cy = 0.20, 0.20, 0.05
                else:
                    endConst = 30
                    cKp, cKd, cy = 0.25, 0.25, 0.08

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

                # Emergency Reversing Safety Check (Blocked directly by pillar)
                if ((cPillar.area > 6500 and cPillar.target == redTarget) or 
                    (cPillar.area > 8000 and cPillar.target == greenTarget)) and cPillar.y > 350 and not tempParking:
                    print("[SAFETY] Dangerously close to pillar! Executing emergency reverse...")
                    serial_ctrl.send_command("BACKWARD")
                    time.sleep(0.6)
                    serial_ctrl.send_command("FORWARD")
                    continue

                # Constrain angle between safe mechanical limits (60 to 140 deg)
                angle = max(60, min(140, angle))

                # DYNAMIC ANTI-DRIFT SPEED CONTROL:
                # When steer angle deflection > 30° (angle < 70 or angle > 130), reduce speed to 195!
                steerDeflection = abs(angle - 100)
                currentSpeed = turnSpeed if steerDeflection > 30 else normalSpeed

                # Rate-limiting: Send DRIVE command only when angle/speed changes or every 100ms
                angle_changed = last_steer_angle is None or abs(angle - last_steer_angle) >= 2
                speed_changed = last_drive_speed != currentSpeed
                time_elapsed = (currTime - last_cmd_time) >= 0.1

                if angle_changed or speed_changed or time_elapsed:
                    serial_ctrl.send_command("AUTO_US_OFF")
                    serial_ctrl.send_command(f"DRIVE:{currentSpeed}:{angle}")
                    last_steer_angle = angle
                    last_drive_speed = currentSpeed
                    last_cmd_time = currTime

            # -------------------------------------------------------------
            # 4. FINAL LAP MAGENTA PARKING LOT ALGORITHM (t >= 12)
            # -------------------------------------------------------------
            if t >= 12 and not isTurning and not tempParking:
                print(f"[PARKING] 12 turns (3 laps) complete! Searching for Magenta Parking Lot...")
                tempParking = True

            if tempParking and not isTurning:
                if magentaArea > 3000:
                    navMode = "PARKING_LOT"
                    print("[PARKING] Entering Magenta Parking Lot!")
                    angle = sharpLeft if turnDir == "left" else sharpRight
                    serial_ctrl.send_command(f"DRIVE:{turnSpeed}:{angle}")
                    time.sleep(1.2)
                    serial_ctrl.send_command("STOP")
                    print("[FINISH] Obstacle Challenge Complete!")
                    break

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
            draw_offset_contours(img_disp, contours_orange, ROI4, (0, 165, 255), 2)  # Orange contour for Orange Line
            draw_offset_contours(img_disp, contours_blue, ROI4, (255, 0, 0), 2)      # Blue contour for Blue Line
            draw_offset_contours(img_disp, contours_magenta, ROI4, (255, 0, 255), 2) # Magenta contour for Parking Lot

            cam_type = "WEBCAM" if camera.is_webcam else "PICAM2"
            lock_rem = max(0.0, round(lineLockoutUntil - currTime, 1))
            lock_str = f"LOCKED({lock_rem}s)" if lock_rem > 0 else "READY"
            
            if isTurning:
                t_ela = round(currTime - turnStartTime, 1)
                state_str = f"TURNING ({turnDir.upper()} {t_ela}s)"
            else:
                state_str = f"{navMode} ({turnDir.upper()})"

            active_speed = last_drive_speed if last_drive_speed is not None else normalSpeed
            telemetry_text = f"Cam:{cam_type} | State:{state_str} | Speed:{active_speed} | Turns:{t}/12"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | LineLock:{lock_str}"
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
                elif key == ord('l'):
                    turnDir = "left"
                    print("[KEYBOARD OVERRIDE] Direction set to LEFT")
                elif key == ord('r'):
                    turnDir = "right"
                    print("[KEYBOARD OVERRIDE] Direction set to RIGHT")

            display_variables({
                "Camera Type": cam_type,
                "State": state_str,
                "Track Dir": turnDir,
                "Speed (PWM)": active_speed,
                "Turn Count": f"{t}/12",
                "Line Lockout": lock_str,
                "Line Detected": lDetected,
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
