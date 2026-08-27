#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (1-LAP TEST SCRIPT)

1-Lap Fast Testing Version:
- Performs Return-to-Home after completing exactly 1 LAP (4 turns) instead of 3 laps (12 turns).
- Perfect for quick bench and track testing!
- Uses Electronic Reverse Braking + Slow Approach Speed 175 for pin-point 0cm offset stopping accuracy.
- Corrected Steering Angles: 60° = LEFT TURN, 140° = RIGHT TURN.
"""

import sys
import time
import select
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rOrange, rBlack, rBlue
from wro_functions import (CameraManager, find_black_wall_contours, find_orange_line_contours, find_contours, max_contour, draw_roi,
                           draw_offset_contours, display_variables)


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 1 Open Challenge (1-LAP TEST NODE)")
    print("   Architecture: 1-Lap (4 Turns) Corrected Steering (60=LEFT, 140=RIGHT)")
    print("=" * 65)

    force_webcam = "--webcam" in sys.argv or "-w" in sys.argv
    use_vision_walls = "--vision-walls" in sys.argv or "--no-us" in sys.argv

    forced_dir = "none"
    if "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 < len(sys.argv):
            forced_dir = sys.argv[idx + 1].lower()
            print(f"[CONFIG] Forcing fixed track direction: {forced_dir.upper()}")

    serial_ctrl = WROSerialController()
    if not serial_ctrl.connect():
        print("[ERROR] Cannot proceed without ESP32 serial connection.")
        sys.exit(1)

    print("[SAFETY] Forcing robot STOP state during initialization...")
    serial_ctrl.send_command("STOP")
    time.sleep(0.5)

    show_monitor_display = "--no-display" not in sys.argv
    window_name = "WRO Open Challenge - 1-Lap Test Monitor Debug (Pi 5)"

    if show_monitor_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            print("[DISPLAY] Created OpenCV live display window on monitor!")
        except Exception as e:
            print(f"[WARNING] Could not open GUI display window: {e}")
            show_monitor_display = False

    camera = CameraManager(force_webcam=force_webcam, device_index=0)
    camera.start()

    print("[INFO] Capturing camera warmup frames...")
    for _ in range(15):
        warmup_frame = camera.capture_array()
        if warmup_frame is not None:
            if show_monitor_display:
                cv2.imshow(window_name, warmup_frame)
                cv2.waitKey(1)
        time.sleep(0.04)

    print("\n[READY] 1-Lap Test Engine Ready!")
    print("[PHASE 1] Recording Baseline Start Position...")

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

    print(f"[START SNAPSHOT] Baseline Position Snapshot Recorded: {start_snapshot}")

    print("[START] Driving FORWARD (1-Lap Test Mode: Target = 4 Turns / 1 Lap)!")
    serial_ctrl.send_command("AUTO_US_OFF")
    serial_ctrl.send_command("FORWARD")

    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [200, 300, 440, 350]  # Ground indicator line ROI

    TARGET_TURNS = 4       # 1 LAP TEST TARGET (4 turns = 1 lap)
    t = 0                  # Completed turn count
    turnDir = forced_dir   # Track direction ("left", "right", or "none")
    lDetected = False
    isTurning = False
    turnStartTime = 0
    lineLockoutUntil = 0   # 3.5s line detection lockout timer
    turnCooldownUntil = 0  # 3.5s turn trigger cooldown timer
    reverseCooldownUntil = 0 # 1.2s emergency reverse cooldown timer
    lockoutDuration = 3.5  # Exactly 3.5 seconds lockout

    normalSpeed = 245      # Full straightaway speed (96% PWM)
    turnSpeed = 195        # Reduced turn & cornering speed to prevent drifting when steering > 30 deg or max 40 deg
    returnSpeed = 230      # Controlled approach speed for pin-point finish stopping (230 PWM)

    minTurnDuration = 0.8  # Minimum arc turn time before checking wall re-acquisition (0.8s)
    maxTurnDuration = 2.2  # Safety maximum turn time cap (2.2s)
    wallReacquireArea = 600 # Area threshold to confirm single wall in narrow FOV view
    turnThresh = 200       # Area threshold below which wall end is detected

    is_returning_home = False          # True once final corner exit is confirmed
    corner_exit_time = 0               # Timestamp when 4th turn exit was confirmed
    home_stop_initiated = False        # True once final stop sequence is committed

    MIN_CLEAR_OF_CORNER_TIME = 0.5     # Min time after turn exit before allowing line/sensor stop (0.5s)
    FRONT_WALL_HARD_STOP_CM = 25.0     # Hard safety ceiling: stop if front wall <= 25cm
    HOME_ABSOLUTE_TIMEOUT = 4.0        # Absolute maximum timeout cap since turn exit

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
            cListOrange = find_orange_line_contours(img, ROI3)
            cListBlue = find_contours(img_lab, rBlue, ROI3)

            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]
            orangeArea = max_contour(cListOrange, ROI3)[0]
            blueArea = max_contour(cListBlue, ROI3)[0]

            # Get latest continuous ultrasonic sensor telemetry from ESP32
            us_data = serial_ctrl.get_us_data()
            f_us = us_data.get("f", 0)
            f1_us = us_data.get("f1", f_us)
            f2_us = us_data.get("f2", f_us)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

            # -------------------------------------------------------------
            # 0. EMERGENCY ANGLED REVERSE FOR INNER & FRONT WALL COLLISIONS
            # -------------------------------------------------------------
            hit_right_inner = (turnDir == "right" and (0 < f2_us <= 13 or 0 < r_us <= 7 or ((f2_us == 0 or r_us == 0) and rightArea > 1300))) or \
                              (0 < r_us <= 6) or (0 < f2_us <= 11 and rightArea > leftArea)
            hit_left_inner  = (turnDir == "left" and (0 < f1_us <= 13 or 0 < l_us <= 7 or ((f1_us == 0 or l_us == 0) and leftArea > 1300))) or \
                              (0 < l_us <= 6) or (0 < f1_us <= 11 and leftArea > rightArea)
            hit_front_wall  = (0 < f_us <= 12) or ((0 < f1_us <= 12) and (0 < f2_us <= 12)) or \
                              ((f_us == 0 or f1_us == 0 or f2_us == 0) and (leftArea > 1500 or rightArea > 1500))

            if (hit_right_inner or hit_left_inner or hit_front_wall) and currTime >= reverseCooldownUntil and not home_stop_initiated:
                if hit_right_inner:
                    rev_steer = 65   # Steer LEFT in reverse -> pulls front nose away from RIGHT/INNER wall
                    wall_name = "INNER/RIGHT WALL"
                elif hit_left_inner:
                    rev_steer = 135  # Steer RIGHT in reverse -> pulls front nose away from LEFT/INNER wall
                    wall_name = "INNER/LEFT WALL"
                else:
                    rev_steer = 65 if turnDir == "right" else 135
                    wall_name = "FRONT WALL"

                print("=" * 65)
                print(f"[EMERGENCY REVERSE] Collision detected with {wall_name}!")
                print(f"[SENSORS] F:{f_us} F1:{f1_us} F2:{f2_us} L:{l_us} R:{r_us} B:{b_us} | Vision L:{leftArea}px R:{rightArea}px")
                print(f"[REVERSE ACTION] Reversing with angle {rev_steer}° to free bot from wall...")
                print("=" * 65)

                serial_ctrl.send_command("STOP")
                time.sleep(0.04)

                rev_start = time.time()
                while True:
                    serial_ctrl.send_command(f"DRIVE:-235:{rev_steer}")
                    time.sleep(0.05)
                    us_check = serial_ctrl.get_us_data()
                    curr_f = us_check.get("f", 0)
                    curr_f1 = us_check.get("f1", curr_f)
                    curr_f2 = us_check.get("f2", curr_f)
                    curr_b = us_check.get("b", 0)

                    rev_elapsed = time.time() - rev_start

                    # Rear collision safety guard
                    if curr_b > 0 and curr_b <= 8:
                        print(f"[SAFETY] Rear wall proximity ({curr_b}cm)! Stopping reverse.")
                        break

                    # Clearance check: front sensors must see open track (>= 18cm or no obstacle)
                    front_cleared = (curr_f >= 18 or curr_f == 0) and \
                                    (curr_f1 >= 18 or curr_f1 == 0) and \
                                    (curr_f2 >= 18 or curr_f2 == 0)

                    if rev_elapsed >= 0.45 and front_cleared:
                        break
                    if rev_elapsed >= 1.2:  # Safety timeout cap
                        break

                serial_ctrl.send_command("STOP")
                time.sleep(0.05)

                # If collided during an active turn, reset turn state so vision re-acquires the lane cleanly
                if isTurning:
                    isTurning = False
                    turnCooldownUntil = time.time() + 0.5

                serial_ctrl.send_command("FORWARD")
                reverseCooldownUntil = time.time() + 1.2
                last_cmd_time = time.time()
                last_drive_speed = normalSpeed
                last_steer_angle = 100
                continue

            # -------------------------------------------------------------
            # 1. PERMANENT FIRST-COLOR DIRECTION LOCK & MARKER DETECTION
            # -------------------------------------------------------------
            if t < TARGET_TURNS and not isTurning and not is_returning_home and currTime >= lineLockoutUntil:
                if turnDir == "none":
                    if orangeArea > 100 and orangeArea > blueArea:
                        turnDir = "right"
                        lDetected = True
                        lineLockoutUntil = currTime + 1.2
                        print(f"[FIRST-COLOR LOCK] First Line Detected: ORANGE ({orangeArea} px) -> Permanently Locking Direction to RIGHT!")
                    elif blueArea > 100 and blueArea > orangeArea:
                        turnDir = "left"
                        lDetected = True
                        lineLockoutUntil = currTime + 1.2
                        print(f"[FIRST-COLOR LOCK] First Line Detected: BLUE ({blueArea} px) -> Permanently Locking Direction to LEFT!")
                
                elif turnDir == "right":
                    if orangeArea > 100:
                        lDetected = True
                        lineLockoutUntil = currTime + 1.2
                        print(f"[LOCKED MARKER] Detected ORANGE Line ({orangeArea} px) -> Track Dir = RIGHT")
                
                elif turnDir == "left":
                    if blueArea > 100:
                        lDetected = True
                        lineLockoutUntil = currTime + 1.2
                        print(f"[LOCKED MARKER] Detected BLUE Line ({blueArea} px) -> Track Dir = LEFT")

            # -------------------------------------------------------------
            # 2. HYBRID CORNER TURN & DYNAMIC VISION EXIT (60=LEFT, 140=RIGHT)
            # -------------------------------------------------------------
            if isTurning and not is_returning_home:
                targetTurnAngle = 60 if turnDir == "left" else 140
                
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
                    turnCooldownUntil = currTime + 0.8
                    lineLockoutUntil = currTime + 0.8
                    exit_reason = "WALL_REACQUIRED" if newWallAcquired else "MAX_TIMEOUT"
                    print(f"[NAV EVENT] Turn {t}/{TARGET_TURNS} ({turnDir.upper()}) EXITED via {exit_reason} in {round(turnElapsed, 2)}s!")

                    if t >= TARGET_TURNS:
                        is_returning_home = True
                        corner_exit_time = currTime
                        serial_ctrl.send_command("AUTO_US_ON")
                        print("=" * 65)
                        print(f"[PHASE 3] 1 LAP COMPLETE ({t}/{TARGET_TURNS} Turns)! Approaching Start/Finish Line at returnSpeed {returnSpeed}...")
                        print("[PHASE 3] Ultrasonics & Finish Line Scanner Active.")
                        print("=" * 65)

            elif (t < TARGET_TURNS) and currTime >= turnCooldownUntil and not is_returning_home:
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

                # Trigger turn if wall dropped and direction is locked (or marker was seen)
                if (lDetected or turnDir != "none" or forced_dir != "none") and wallDropDetected:
                    targetTurnAngle = 60 if turnDir == "left" else 140
                    t += 1
                    marker_info = "Marker + Wall Drop" if lDetected else "Inner Wall Drop"
                    print(f"[NAV EVENT] {marker_info}! (L:{leftArea} R:{rightArea}) -> Triggering Turn ({t}/{TARGET_TURNS}) angle={targetTurnAngle}...")
                    serial_ctrl.send_command(f"DRIVE:{turnSpeed}:{targetTurnAngle}")
                    last_cmd_time = currTime
                    last_drive_speed = turnSpeed
                    last_steer_angle = targetTurnAngle
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False
                    turnCooldownUntil = currTime + maxTurnDuration + 0.8

            # -------------------------------------------------------------
            # 3. STRAIGHTAWAY WALL AVOIDANCE & DYNAMIC SPEED CONTROL (1-Lap Run)
            # -------------------------------------------------------------
            if not isTurning and not is_returning_home:
                # Check whether both walls or single wall is in view
                both_walls_visible = (leftArea > 250 and rightArea > 250)

                if both_walls_visible:
                    # Dual-wall proportional centering
                    aDiff = rightArea - leftArea
                    steer_angle = int(100 - (aDiff * 0.015))
                elif turnDir == "right" and rightArea <= 250:
                    # Approaching RIGHT turn: Right (inner) wall dropped.
                    # Do NOT steer hard right into inner wall! Maintain straight course using outer left wall.
                    left_err = leftArea - 900
                    steer_angle = int(100 - (left_err * 0.008))
                    steer_angle = max(90, min(110, steer_angle))
                elif turnDir == "left" and leftArea <= 250:
                    # Approaching LEFT turn: Left (inner) wall dropped.
                    # Do NOT steer hard left into inner wall! Maintain straight course using outer right wall.
                    right_err = rightArea - 900
                    steer_angle = int(100 + (right_err * 0.008))
                    steer_angle = max(90, min(110, steer_angle))
                else:
                    # Single wall fallback
                    aDiff = rightArea - leftArea
                    steer_angle = int(100 - (aDiff * 0.01))

                # Sensor side-proximity safety nudges
                if 0 < r_us <= 12:
                    steer_angle = min(steer_angle, 80) # Nudge left away from right wall
                elif 0 < l_us <= 12:
                    steer_angle = max(steer_angle, 120) # Nudge right away from left wall

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
            # 4. Phase 3: PRECISION STARTING SECTION STOPPING ENGINE (1-Lap Test)
            # -------------------------------------------------------------
            if is_returning_home and not home_stop_initiated:
                elapsed_since_corner = currTime - corner_exit_time

                aDiff = rightArea - leftArea
                gentle_steer = int(100 - (aDiff * 0.01))
                gentle_steer = max(80, min(120, gentle_steer))

                reasons = []

                line_marker_detected = (
                    (turnDir == "right" and orangeArea > 150) or
                    (turnDir == "left" and blueArea > 150) or
                    (turnDir == "none" and (orangeArea > 150 or blueArea > 150))
                )
                if elapsed_since_corner >= MIN_CLEAR_OF_CORNER_TIME and line_marker_detected:
                    reasons.append(f"START_FINISH_LINE_MARKER(O:{orangeArea},B:{blueArea})")

                init_b = start_snapshot.get("b", 0)
                init_f = start_snapshot.get("f", 0)
                b_match = (init_b > 0 and b_us > 0 and abs(b_us - init_b) <= 4.0)
                f_match = (init_f > 0 and f_us > 0 and abs(f_us - init_f) <= 4.0)
                if elapsed_since_corner >= MIN_CLEAR_OF_CORNER_TIME and (b_match or f_match):
                    reasons.append(f"BASELINE_SENSOR_MATCH(F:{f_us}/{init_f},B:{b_us}/{init_b})")

                if f_us > 0 and f_us <= FRONT_WALL_HARD_STOP_CM:
                    reasons.append(f"FRONT_WALL_PROXIMITY({f_us}cm)")

                if elapsed_since_corner >= HOME_ABSOLUTE_TIMEOUT:
                    reasons.append("SAFETY_TIMEOUT_CAP")

                should_stop_now = len(reasons) > 0

                if should_stop_now:
                    home_stop_initiated = True
                    print("=" * 65)
                    print(f"[FINISH PRECISION STOP] Executing Active Electronic Reverse Brake Pulse!")
                    print(f"[FINISH METRICS] Reasons: {reasons}")
                    print(f"[FINISH METRICS] Elapsed since Turn 4 exit: {round(elapsed_since_corner, 2)}s")
                    print(f"[SENSORS] Current F:{f_us} L:{l_us} R:{r_us} B:{b_us} | Baseline: {start_snapshot}")
                    print("=" * 65)

                    serial_ctrl.send_command("STOP")
                    serial_ctrl.send_command("DRIVE:-180:100")
                    time.sleep(0.08)
                    serial_ctrl.send_command("STOP")
                    last_drive_speed = 0
                    time.sleep(0.5)
                    break
                else:
                    if (currTime - last_cmd_time) >= 0.1 or last_drive_speed != returnSpeed:
                        serial_ctrl.send_command(f"DRIVE:{returnSpeed}:{gentle_steer}")
                        last_cmd_time = currTime
                        last_drive_speed = returnSpeed
                        last_steer_angle = gentle_steer

            elif home_stop_initiated:
                serial_ctrl.send_command("STOP")
                last_drive_speed = 0
                time.sleep(0.5)
                break

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
                state_str = f"RETURN_TO_HOME_1LAP ({returnSpeed})"
            elif isTurning:
                t_ela = round(currTime - turnStartTime, 1)
                state_str = f"TURNING ({turnDir.upper()} {t_ela}s)"
            else:
                state_str = f"VISION_WALLS ({turnDir.upper()})"
            
            active_speed = last_drive_speed if last_drive_speed is not None else normalSpeed
            telemetry_text = f"Cam:{cam_type} | State:{state_str} | Speed:{active_speed} | Turns:{t}/{TARGET_TURNS}"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | LineLock:{lock_str}"
            us_text = f"US -> F:{f_us} F1:{f1_us} F2:{f2_us} | L:{l_us} R:{r_us} B:{b_us}"

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
                "Turn Count": f"{t}/{TARGET_TURNS}",
                "Line Lockout": lock_str,
                "Line Detected": lDetected,
                "Left Wall Area (px)": leftArea,
                "Right Wall Area (px)": rightArea,
                "US Front (cm)": f_us,
                "US Front1 (cm)": f1_us,
                "US Front2 (cm)": f2_us,
                "US Left (cm)": l_us,
                "US Right (cm)": r_us,
                "US Back (cm)": b_us,
                "Home Baseline": start_snapshot
            })

            time.sleep(0.02)

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
