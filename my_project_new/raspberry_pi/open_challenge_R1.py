#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Hybrid Sensor-Vision Control Architecture:
- Position Tracking & Home Baseline Snapshot: Recorded during warm-up countdown (f1, f2, L, R, B).
- Duty Cycle Optimization for Ultrasonic Sensors:
  * Phase 1 (Warmup): Active to capture home position snapshot.
  * Phase 2 (Laps 1-3): Deactivated (AUTO_US_OFF) during active vision navigation to eliminate lag.
  * Phase 3 (Post-3-Lap Return): Re-activated after 5-second post-turn-12 delay to detect home arrival.
- Post-3-Lap Return to Home Behavior (with 5-Second Post-Turn-12 Delay):
  * After 12th turn completes, robot drives 5.0 seconds forward down home stretch.
  * FORWARD ONLY (No backward driving). Continues driving forward along closed circuit track.
  * Reduced Set Speed: 230 (global turnSpeed = 230).
  * Priority on Stopping: Dampened camera steering (no violent wall corrections).
  * Stopping Accuracy: Halts bot completely when sensors match initial snapshot within 2cm buffer.
- Dynamic Anti-Drift Speed Control: Reduces speed to turnSpeed (230) on sharp turns (>30° deflection).
- Permanent First-Color Direction Lock: Locks onto whichever marker color (Orange/Blue) is detected first.
- Vision-Dynamic Corner Exit: Dynamically exits corner turns when camera re-acquires new straightaway wall.
"""

import sys
import time
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rOrange, rBlack, rBlue
from wro_functions import (CameraManager, find_black_wall_contours, find_contours, max_contour, draw_roi,
                           draw_offset_contours, display_variables)


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 1 Open Challenge Node (Pi 5)")
    print("   Architecture: 5s Post-Turn-12 Delay + Return to Home (Speed 230)")
    print("=" * 65)

    force_webcam = "--webcam" in sys.argv or "-w" in sys.argv
    use_vision_walls = "--vision-walls" in sys.argv or "--no-us" in sys.argv

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

    # Force immediate STOP during startup
    print("[SAFETY] Forcing robot STOP state during initialization...")
    serial_ctrl.send_command("STOP")
    time.sleep(0.5)

    show_monitor_display = "--no-display" not in sys.argv
    window_name = "WRO Open Challenge - Hybrid Monitor Debug (Pi 5)"

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

    # ------------------------------------------------------------------------
    # Phase 1: Setup & Warmup Countdown (Record Initial Start Position Snapshot)
    # ------------------------------------------------------------------------
    print("\n[READY] Sensor-Vision Engine Ready!")
    print("[PHASE 1] Recording Baseline Start Position (f1, f2, L, R, B)...")
    
    start_snapshot = {"f": 0, "f1": 0, "f2": 0, "l": 0, "r": 0, "b": 0}
    
    print("[COUNTDOWN] Bot starts driving in 3 seconds... (Press 'q' to abort, 'l'/'r' to set dir)")
    for c in range(3, 0, -1):
        print(f"[COUNTDOWN] {c}...")
        
        # Read baseline sensor values during countdown
        us_data = serial_ctrl.get_us_data()
        f_val = us_data.get("f", 0)
        start_snapshot = {
            "f": f_val,
            "f1": us_data.get("f1", f_val),
            "f2": us_data.get("f2", f_val),
            "l": us_data.get("l", 0),
            "r": us_data.get("r", 0),
            "b": us_data.get("b", 0)
        }

        if show_monitor_display and warmup_frame is not None:
            cd_img = warmup_frame.copy()
            cv2.putText(cd_img, f"STARTING IN {c} SECONDS...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow(window_name, cd_img)
            cv2.waitKey(1)
        time.sleep(1.0)

    print(f"[START SNAPSHOT] Baseline Home Position Recorded: {start_snapshot}")

    # ------------------------------------------------------------------------
    # Phase 2: Start Active Driving (Ultrasonic Sensors Deactivated during Laps)
    # ------------------------------------------------------------------------
    print("[START] Driving FORWARD (Phase 2: Duty Cycle - Ultrasonics Deactivated for Zero-Lag Vision)!")
    serial_ctrl.send_command("AUTO_US_OFF")
    serial_ctrl.send_command("FORWARD")

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [200, 300, 440, 350]  # Ground indicator line ROI

    # Navigation flags & state counters
    t = 0                  # Completed turn count (3 laps x 4 turns = 12)
    turnDir = forced_dir   # Track direction ("left", "right", or "none")
    lDetected = False
    isTurning = False
    is_returning_home = False
    turn12_finish_time = 0 # Timestamp when 12th turn finishes
    return_start_time = 0
    turnStartTime = 0
    lineLockoutUntil = 0   # 3.5s line detection lockout timer
    turnCooldownUntil = 0  # 3.5s turn trigger cooldown timer
    lockoutDuration = 3.5  # Exactly 3.5 seconds lockout
    
    # Speed Parameters (Global turnSpeed updated to 230)
    normalSpeed = 245      # Full straightaway speed (96% PWM)
    turnSpeed = 230        # Global turn & cornering speed (updated to 230 across script)
    returnSpeed = 230      # Set speed for final forward return segment (230)

    # Dynamic Turn Exit Timings (Optimized for Narrow FOV Camera)
    minTurnDuration = 0.8  # Minimum arc turn time before checking wall re-acquisition (0.8s)
    maxTurnDuration = 2.2  # Safety maximum turn time cap (2.2s)
    wallReacquireArea = 600 # Area threshold to confirm single wall in narrow FOV view
    turnThresh = 200       # Area threshold below which wall end is detected

    # Serial rate-limiting variables
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

            # Find contours using dual-layer HSV+LAB black wall segmentation (100% blue exclusion)
            cListLeft = find_black_wall_contours(img, ROI1)
            cListRight = find_black_wall_contours(img, ROI2)
            cListOrange = find_contours(img_lab, rOrange, ROI3)
            cListBlue = find_contours(img_lab, rBlue, ROI3)

            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]
            orangeArea = max_contour(cListOrange, ROI3)[0]
            blueArea = max_contour(cListBlue, ROI3)[0]

            # Sensor readings (Phase 2: US sensors deactivated during active laps; reactivated in Phase 3)
            us_data = serial_ctrl.get_us_data() if is_returning_home else {}
            f_us = us_data.get("f", 0)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

            # -------------------------------------------------------------
            # 1. PERMANENT FIRST-COLOR DIRECTION LOCK & MARKER DETECTION
            # -------------------------------------------------------------
            if not isTurning and not is_returning_home and currTime >= lineLockoutUntil:
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
            # 2. HYBRID CORNER TURN & DYNAMIC VISION EXIT (turnSpeed = 230)
            # -------------------------------------------------------------
            if isTurning and not is_returning_home:
                targetTurnAngle = 140 if turnDir == "left" else 60
                
                # Stream active corner turn at set turnSpeed (230)
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
            
            elif currTime >= turnCooldownUntil and not is_returning_home:
                # Wall drop check (wall area drops below turnThresh)
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

                # STRICT TRIGGER: Require line marker detection (or forced dir) AND wall drop!
                if (lDetected or forced_dir != "none") and wallDropDetected:
                    targetTurnAngle = 140 if turnDir == "left" else 60
                    t += 1
                    print(f"[NAV EVENT] Marker Seen + Wall Drop! (L:{leftArea} R:{rightArea}) -> Triggering Turn ({t}/12) at turnSpeed={turnSpeed}...")
                    serial_ctrl.send_command(f"DRIVE:{turnSpeed}:{targetTurnAngle}")
                    last_cmd_time = currTime
                    last_drive_speed = turnSpeed
                    last_steer_angle = targetTurnAngle
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False  # Reset marker flag for next straightaway!
                    turnCooldownUntil = currTime + maxTurnDuration + lockoutDuration

            # -------------------------------------------------------------
            # 3. STRAIGHTAWAY WALL AVOIDANCE & DYNAMIC SPEED CONTROL
            # -------------------------------------------------------------
            if not isTurning and not is_returning_home:
                # Phase 2: Ultrasonics remain deactivated during 3 laps
                serial_ctrl.send_command("AUTO_US_OFF")

                aDiff = rightArea - leftArea  # Negative when close to left wall
                steer_angle = int(100 - (aDiff * 0.02))
                steer_angle = max(60, min(140, steer_angle))

                # DYNAMIC SPEED CONTROL:
                # When steer angle deflection > 30° (angle < 70 or angle > 130), reduce speed to turnSpeed (230)!
                steerDeflection = abs(steer_angle - 100)
                currentSpeed = turnSpeed if steerDeflection > 30 else normalSpeed

                # Rate-limiting: Send DRIVE command only when angle/speed changes or every 100ms
                angle_changed = last_steer_angle is None or abs(steer_angle - last_steer_angle) >= 2
                speed_changed = last_drive_speed != currentSpeed
                time_elapsed = (currTime - last_cmd_time) >= 0.1

                if angle_changed or speed_changed or time_elapsed:
                    serial_ctrl.send_command(f"DRIVE:{currentSpeed}:{steer_angle}")
                    last_steer_angle = steer_angle
                    last_drive_speed = currentSpeed
                    last_cmd_time = currTime

            # -------------------------------------------------------------
            # Phase 3: POST-3-LAP RETURN TO HOME (ACTIVATES 5.0 SECONDS AFTER TURN 12)
            # -------------------------------------------------------------
            if t >= 12 and not isTurning:
                if turn12_finish_time == 0:
                    turn12_finish_time = currTime
                    print("=" * 65)
                    print("[COMPLETION] 3 Laps (12 Turns) Complete! Driving 5.0s down home stretch before Return-to-Home activation...")
                    print("=" * 65)

                time_since_turn12 = currTime - turn12_finish_time

                # Activate Return-to-Home Phase 3 after 5.0 seconds delay
                if time_since_turn12 >= 5.0:
                    if not is_returning_home:
                        print("=" * 65)
                        print("[PHASE 3] 5.0s Post-Turn Delay Passed! Activating Return to Home...")
                        print("[CONSTRAINT A] FORWARD ONLY (Continuing forward along home stretch).")
                        print(f"[CONSTRAINT B] Smoothly reducing speed to set returnSpeed = {returnSpeed} (230).")
                        print("[CONSTRAINT C] Priority #1: Stopping accuracy. Dampening camera steering to maintain stable drive.")
                        print("[CONSTRAINT D] Phase 3 Duty Cycle: Re-activating Ultrasonic Sensors to detect 2cm Home buffer...")
                        print("=" * 65)
                        is_returning_home = True
                        return_start_time = currTime
                        serial_ctrl.send_command("AUTO_US_ON")  # Phase 3: Re-activate ultrasonic sensors for home detection

                    return_elapsed = currTime - return_start_time
                    
                    # Check sensor match with initial warm-up snapshot (within 2cm buffer)
                    init_b = start_snapshot.get("b", 0)
                    init_f = start_snapshot.get("f", 0)
                    init_l = start_snapshot.get("l", 0)
                    init_r = start_snapshot.get("r", 0)

                    b_match = (init_b > 0 and b_us > 0 and abs(b_us - init_b) <= 2.0)
                    f_match = (init_f > 0 and f_us > 0 and abs(f_us - init_f) <= 2.0)
                    side_match = (init_l > 0 and l_us > 0 and abs(l_us - init_l) <= 2.0) and (init_r > 0 and r_us > 0 and abs(r_us - init_r) <= 2.0)
                    
                    # Stopping accurately at recorded start position is Priority #1!
                    home_reached = (b_match or f_match or side_match or return_elapsed >= 4.0)

                    if home_reached and return_elapsed >= 0.5:
                        print("=" * 65)
                        print(f"[FINISH] ACCURATELY STOPPED AT RECORDED HOME POSITION IN {round(return_elapsed, 2)}s!")
                        print(f"[HOME METRICS] Current Sensors: F:{f_us} L:{l_us} R:{r_us} B:{b_us}")
                        print(f"[BASELINE SNAPSHOT] Start Snapshot: {start_snapshot}")
                        print("=" * 65)
                        serial_ctrl.send_command("STOP")
                        time.sleep(0.5)
                        break
                    else:
                        # Constraint C: Heavily dampened, stable steering (NO violent camera steering!)
                        aDiff = rightArea - leftArea
                        gentle_steer = int(100 - (aDiff * 0.005))   # Heavily dampened gain 0.005
                        gentle_steer = max(85, min(115, gentle_steer)) # Strict stable bounds [85, 115]

                        # Constraint A & B: Forward drive at slow set speed 230
                        if (currTime - last_cmd_time) >= 0.1 or last_drive_speed != returnSpeed:
                            serial_ctrl.send_command(f"DRIVE:{returnSpeed}:{gentle_steer}")
                            last_cmd_time = currTime
                            last_drive_speed = returnSpeed
                            last_steer_angle = gentle_steer

            # Draw ROIs & Offset Contours (matching my_old_contour_colorvals_crt.py)
            img_disp = img.copy()
            draw_roi(img_disp, ROI1, (0, 255, 255), 2)
            draw_roi(img_disp, ROI2, (0, 255, 255), 2)
            draw_roi(img_disp, ROI3, (255, 255, 0), 2)
            draw_offset_contours(img_disp, cListLeft, ROI1, (0, 255, 0), 2)
            draw_offset_contours(img_disp, cListRight, ROI2, (0, 255, 0), 2)
            draw_offset_contours(img_disp, cListOrange, ROI3, (0, 165, 255), 2)
            draw_offset_contours(img_disp, cListBlue, ROI3, (255, 0, 0), 2)

            cam_type = "WEBCAM" if camera.is_webcam else "PICAM2"
            lock_rem = max(0.0, round(lineLockoutUntil - currTime, 1))
            lock_str = f"LOCKED({lock_rem}s)" if lock_rem > 0 else "READY"
            
            if is_returning_home:
                state_str = f"RETURN_TO_HOME ({returnSpeed})"
            elif t >= 12 and not isTurning:
                t_del = round(currTime - turn12_finish_time, 1)
                state_str = f"HOME_STRETCH ({t_del}s/5.0s)"
            elif isTurning:
                t_ela = round(currTime - turnStartTime, 1)
                state_str = f"TURNING ({turnDir.upper()} {t_ela}s)"
            else:
                state_str = f"VISION_WALLS ({turnDir.upper()})"
            
            active_speed = last_drive_speed if last_drive_speed is not None else normalSpeed
            telemetry_text = f"Cam:{cam_type} | State:{state_str} | Speed:{active_speed} | Turns:{t}/12"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | LineLock:{lock_str}"
            us_text = f"US Sensors -> F:{f_us}cm | L:{l_us}cm | R:{r_us}cm | B:{b_us}cm"

            cv2.putText(img_disp, telemetry_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 204), 2)
            cv2.putText(img_disp, wall_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(img_disp, us_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Display directly on monitor screen & keyboard controls
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
                "US Front (cm)": f_us,
                "US Left (cm)": l_us,
                "US Right (cm)": r_us,
                "US Back (cm)": b_us,
                "Home Snapshot": start_snapshot
            })

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
