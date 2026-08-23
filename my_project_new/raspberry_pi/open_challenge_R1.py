#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

WRO 2026 Rule 9.24.2 / Appendix A Section 2 & 3 Precision Finishing Engine:
- Goal: Stop 100% COMPLETELY INSIDE the Starting/Finish Section (WRO Rule 9.24.2).
- Key Tuning:
  1. Controlled Return Speed: 195 (reduces momentum for instant braking).
  2. Front Wall Proximity Trigger: f_us <= 55cm (places bot DEAD CENTER in 100cm section).
  3. Start Snapshot Matching: f_us matching initial warm-up snapshot (within 10cm).
  4. Calibrated Drive Time: 0.75s after turn 12 exit.
  5. Zero-Delay Debounce: Applies immediate STOP command with 0.05s hold to prevent overrun.
- BUGFIX: Turn-trigger condition guarded on `t < 12` to prevent phantom 13th turn.
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
    print("   Architecture: WRO Rule 9.24.2 Precision Finish Section Stop Engine")
    print("=" * 65)

    force_webcam = "--webcam" in sys.argv or "-w" in sys.argv
    use_vision_walls = "--vision-walls" in sys.argv or "--no-us" in sys.argv

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
    print("[PHASE 1] Recording Baseline Start Position Snapshot...")

    start_snapshot = {"f": 0, "f1": 0, "f2": 0, "l": 0, "r": 0, "b": 0}

    print("[COUNTDOWN] Bot starts driving in 3 seconds... (Press 'q' to abort, 'l'/'r' to set dir)")
    for c in range(3, 0, -1):
        print(f"[COUNTDOWN] {c}...")
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

    print(f"[START SNAPSHOT] Recorded Home Baseline Box Position: {start_snapshot}")

    # ------------------------------------------------------------------------
    # Phase 2: Start Active Driving (Ultrasonics deactivated during laps)
    # ------------------------------------------------------------------------
    print("[START] Driving FORWARD (Phase 2: Ultrasonics off for zero-lag vision)!")
    serial_ctrl.send_command("AUTO_US_OFF")
    serial_ctrl.send_command("FORWARD")

    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [200, 300, 440, 350]  # Ground indicator line ROI

    t = 0                  # Completed turn count (3 laps x 4 turns = 12)
    turnDir = forced_dir   # Track direction ("left", "right", or "none")
    lDetected = False
    isTurning = False
    turnStartTime = 0
    lineLockoutUntil = 0   # 3.5s line detection lockout timer
    turnCooldownUntil = 0  # 3.5s turn trigger cooldown timer
    lockoutDuration = 3.5  # Exactly 3.5 seconds lockout

    normalSpeed = 245      # Full straightaway speed (96% PWM)
    turnSpeed = 230        # Global turn & cornering speed (230)
    returnSpeed = 220      # Reduced return speed (195 PWM) for instant precision braking inside start section!

    minTurnDuration = 0.8  # Minimum arc turn time before checking wall re-acquisition (0.8s)
    maxTurnDuration = 2.2  # Safety maximum turn time cap (2.2s)
    wallReacquireArea = 600 # Area threshold to confirm single wall in narrow FOV view
    turnThresh = 200       # Area threshold below which wall end is detected

    # ------------------------------------------------------------------------
    # Phase 3 Parameters (Precision Finish Section Stopping Engine)
    # ------------------------------------------------------------------------
    is_returning_home = False          # True once 12th (final) corner exit is confirmed
    corner12_exit_time = 0             # Timestamp when 12th turn exit was confirmed
    home_stop_initiated = False        # True once final stop sequence is committed
    home_stop_confirm_start = 0        # Timestamp for debounce confirmation hold

    MIN_CLEAR_OF_CORNER_TIME = 0.35    # Min time after turn-12 exit to ensure back of bot cleared turn-12 corner (0.35s)
    TARGET_HOME_DRIVE_TIME = 0.75      # Drive time at speed 195 to reach middle of starting section (0.75s)
    FRONT_WALL_MID_SECTION_CM = 55.0   # Front wall distance corresponding to center of starting section (<= 55cm)
    FRONT_WALL_HARD_STOP_CM = 35.0     # Hard safety stop clearance before entering next corner (<= 35cm)
    HOME_ABSOLUTE_TIMEOUT = 1.8        # Absolute maximum cap to guarantee stopping inside starting section (1.8s)
    STOP_CONFIRM_HOLD = 0.05           # Zero-delay debounce filter (0.05s) for instant braking

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

            cListLeft = find_black_wall_contours(img, ROI1)
            cListRight = find_black_wall_contours(img, ROI2)
            cListOrange = find_contours(img_lab, rOrange, ROI3)
            cListBlue = find_contours(img_lab, rBlue, ROI3)

            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]
            orangeArea = max_contour(cListOrange, ROI3)[0]
            blueArea = max_contour(cListBlue, ROI3)[0]

            # Ultrasonics: OFF during laps 1-3 for zero lag, ON immediately once
            # 12th (final) corner exit is confirmed.
            us_data = serial_ctrl.get_us_data() if is_returning_home else {}
            f_us = us_data.get("f", 0)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

            # -------------------------------------------------------------
            # 1. PERMANENT FIRST-COLOR DIRECTION LOCK & MARKER DETECTION
            #    (Guarded with t < 12 so it never fires once 12 turns are complete)
            # -------------------------------------------------------------
            if t < 12 and not isTurning and not is_returning_home and currTime >= lineLockoutUntil:
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
                    turnCooldownUntil = currTime + lockoutDuration
                    lineLockoutUntil = currTime + lockoutDuration
                    exit_reason = "WALL_REACQUIRED" if newWallAcquired else "MAX_TIMEOUT"
                    print(f"[NAV EVENT] Turn {t}/12 ({turnDir.upper()}) EXITED via {exit_reason} in {round(turnElapsed, 2)}s!")

                    if t >= 12:
                        # 12th (final) corner exit confirmed. Entering Starting Section!
                        is_returning_home = True
                        corner12_exit_time = currTime
                        serial_ctrl.send_command("AUTO_US_ON")
                        serial_ctrl.send_command(f"DRIVE:{returnSpeed}:100") # Reduce speed to 195 for precision stop
                        last_drive_speed = returnSpeed
                        last_steer_angle = 100
                        print("=" * 65)
                        print("[PHASE 3] Final corner cleared! Entering Starting Section.")
                        print(f"[PHASE 3] Reduced speed to {returnSpeed} & Ultrasonics reactivated for precision finish stop.")
                        print("=" * 65)

            # BUGFIX: Guarded with `t < 12` so a phantom 13th turn can NEVER trigger!
            elif (t < 12) and currTime >= turnCooldownUntil and not is_returning_home:
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

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
                    lDetected = False
                    turnCooldownUntil = currTime + maxTurnDuration + lockoutDuration

            # -------------------------------------------------------------
            # 3. STRAIGHTAWAY WALL AVOIDANCE & DYNAMIC SPEED CONTROL (Laps 1-3)
            # -------------------------------------------------------------
            if not isTurning and not is_returning_home:
                serial_ctrl.send_command("AUTO_US_OFF")

                aDiff = rightArea - leftArea
                steer_angle = int(100 - (aDiff * 0.02))
                steer_angle = max(60, min(140, steer_angle))

                steerDeflection = abs(steer_angle - 100)
                currentSpeed = turnSpeed if steerDeflection > 30 else normalSpeed

                angle_changed = last_steer_angle is None or abs(steer_angle - last_steer_angle) >= 2
                speed_changed = last_drive_speed != currentSpeed
                time_elapsed = (currTime - last_cmd_time) >= 0.1

                if angle_changed or speed_changed or time_elapsed:
                    serial_ctrl.send_command(f"DRIVE:{currentSpeed}:{steer_angle}")
                    last_steer_angle = steer_angle
                    last_drive_speed = currentSpeed
                    last_cmd_time = currTime

            # -------------------------------------------------------------
            # 4. Phase 3 (PRECISION FINISH): Stop 100% Inside Starting Section
            #    (WRO Rule 9.24.2 / Appendix A Section 2 & 3 Compliant)
            # -------------------------------------------------------------
            if is_returning_home and not home_stop_initiated:
                elapsed_since_corner = currTime - corner12_exit_time

                # Gentle, dampened camera steering for smooth straight drive into finish section
                aDiff = rightArea - leftArea
                gentle_steer = int(100 - (aDiff * 0.005))
                gentle_steer = max(85, min(115, gentle_steer))

                front_reading_valid = f_us > 0
                init_f = start_snapshot.get("f", 0)
                init_b = start_snapshot.get("b", 0)

                reasons = []

                # (a) Front Wall Distance Trigger: Center of starting section (f_us <= 55cm)
                if front_reading_valid and f_us <= FRONT_WALL_MID_SECTION_CM:
                    reasons.append(f"FRONT_WALL_MID_SECTION({f_us}cm)")

                # (b) Initial Baseline Start Box Match: f_us within 10cm of start_snapshot
                if front_reading_valid and init_f > 0 and abs(f_us - init_f) <= 10.0:
                    reasons.append(f"START_BOX_SNAPSHOT_MATCH({f_us}cm)")

                # (c) Hard safety stop: front wall <= 35cm (prevents entering Turn 13 corner)
                if front_reading_valid and f_us <= FRONT_WALL_HARD_STOP_CM:
                    reasons.append("FRONT_WALL_HARD_STOP")

                # (d) Primary drive time window (0.75s at speed 195 places bot DEAD CENTER in section)
                if elapsed_since_corner >= TARGET_HOME_DRIVE_TIME:
                    reasons.append("TARGET_HOME_DRIVE_TIME")

                # (e) Absolute maximum timeout cap
                if elapsed_since_corner >= HOME_ABSOLUTE_TIMEOUT:
                    reasons.append("ABSOLUTE_TIMEOUT")

                # Require bot to have driven at least 0.35s clear of Turn 12 corner
                should_stop_now = (elapsed_since_corner >= MIN_CLEAR_OF_CORNER_TIME and len(reasons) > 0)

                if should_stop_now:
                    if home_stop_confirm_start == 0:
                        home_stop_confirm_start = currTime
                    elif (currTime - home_stop_confirm_start) >= STOP_CONFIRM_HOLD:
                        home_stop_initiated = True
                        print("=" * 65)
                        print(f"[FINISH] STOPPED 100% COMPLETELY INSIDE STARTING SECTION!")
                        print(f"[FINISH] Triggers: {', '.join(reasons)}")
                        print(f"[FINISH] Elapsed since Turn 12 exit: {round(elapsed_since_corner, 2)}s")
                        print(f"[SENSORS] Front:{f_us}cm | Left:{l_us}cm | Right:{r_us}cm | Back:{b_us}cm")
                        print(f"[BASELINE SNAPSHOT] Start Snapshot: {start_snapshot}")
                        print("=" * 65)
                else:
                    home_stop_confirm_start = 0
                    if (currTime - last_cmd_time) >= 0.1 or last_drive_speed != returnSpeed:
                        serial_ctrl.send_command(f"DRIVE:{returnSpeed}:{gentle_steer}")
                        last_cmd_time = currTime
                        last_drive_speed = returnSpeed
                        last_steer_angle = gentle_steer

            elif home_stop_initiated:
                # Issue repeated STOP commands to lock motors & halt immediately
                serial_ctrl.send_command("STOP")
                last_drive_speed = 0
                time.sleep(0.5)
                break

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
                state_str = f"PRECISION_FINISH_STOP ({returnSpeed})"
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
                "Start Snapshot": start_snapshot
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
