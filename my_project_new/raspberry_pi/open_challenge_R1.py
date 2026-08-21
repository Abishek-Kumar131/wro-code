#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Hybrid Sensor-Vision Control Architecture:
- Dual-Layer HSV+LAB Black Wall Segmentation with explicit HSV Blue/Orange Mask Subtraction (0% Overlap).
- Vision-Dynamic Corner Exit: Dynamically exits corner turns when the camera
  re-acquires the new straightaway wall (min 0.8s, max 2.2s).
- Decoupled Line Lockout (3.5s) and Turn Cooldown (3.5s) timers.
- Camera Vision Wall Avoidance: Dynamically calculates steering angle to deflect AWAY from black walls.
"""

import sys
import time
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rOrange, rBlack, rBlue
from wro_functions import (CameraManager, find_black_wall_contours, find_contours, max_contour, draw_roi,
                           draw_offset_contours, display_variables)
from camera_streamer import CameraDebugStreamer


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 1 Open Challenge Node (Pi 5)")
    print("   Hybrid Architecture: Vision-Dynamic Corner Turn Exit")
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

    # 2. Start Live Web Camera Debug Streamer (port 8080)
    streamer = CameraDebugStreamer(port=8080)
    streamer.start()

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

    # 4. Safety Countdown before bot starts driving
    print("\n[READY] Hybrid Sensor-Vision Engine Ready!")
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

    # 5. Start robot driving forward
    print("[START] Driving FORWARD with Vision-Dynamic Turn Exit!")
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
    turnStartTime = 0
    lineLockoutUntil = 0   # 3.5s line detection lockout timer
    turnCooldownUntil = 0  # 3.5s turn trigger cooldown timer
    lockoutDuration = 3.5  # Exactly 3.5 seconds lockout
    
    # Dynamic Turn Exit Timings (Optimized for Narrow FOV Camera)
    minTurnDuration = 0.8  # Minimum arc turn time before checking wall re-acquisition (0.8s)
    maxTurnDuration = 2.2  # Safety maximum turn time cap (2.2s)
    wallReacquireArea = 600 # Area threshold to confirm single wall in narrow FOV view
    turnThresh = 200       # Area threshold below which wall end is detected

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

            # Get latest ultrasonic sensor telemetry from ESP32
            us_data = serial_ctrl.get_us_data()
            f_us = us_data.get("f", 0)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)
            us_hardware_working = (l_us > 5 or r_us > 5) and not use_vision_walls

            # -------------------------------------------------------------
            # 1. LINE MARKER DETECTION (WITH DECOUPLED 3.5-SECOND LOCKOUT)
            # -------------------------------------------------------------
            if not isTurning and currTime >= lineLockoutUntil:
                if orangeArea > 150 and orangeArea > blueArea:
                    lDetected = True
                    if forced_dir == "none":
                        turnDir = "right"
                    lineLockoutUntil = currTime + lockoutDuration
                    print(f"[VISION MARKER] Detected ORANGE Line ({orangeArea} px) -> Track Dir = RIGHT (3.5s Line Lockout)")
                elif blueArea > 150 and blueArea > orangeArea:
                    lDetected = True
                    if forced_dir == "none":
                        turnDir = "left"
                    lineLockoutUntil = currTime + lockoutDuration
                    print(f"[VISION MARKER] Detected BLUE Line ({blueArea} px) -> Track Dir = LEFT (3.5s Line Lockout)")

            # -------------------------------------------------------------
            # 2. HYBRID CORNER TURN & DYNAMIC VISION EXIT
            # -------------------------------------------------------------
            if isTurning:
                # Continuously stream active turn command to refresh ESP32 500ms watchdog & lock servo angle!
                targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                serial_ctrl.send_command(targetTurnCmd)

                turnElapsed = currTime - turnStartTime

                # DYNAMIC TURN EXIT CONDITION:
                # After minTurnDuration (0.8s), exit as soon as new wall is acquired (leftArea >= 600 or rightArea >= 600),
                # OR when maxTurnDuration (2.2s) safety timeout is reached!
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
                    targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                    t += 1
                    print(f"[NAV EVENT] Marker Seen + Wall Drop! (L:{leftArea} R:{rightArea}) -> Triggering {targetTurnCmd} ({t}/12)...")
                    serial_ctrl.send_command(targetTurnCmd)
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False  # Reset marker flag for next straightaway!
                    turnCooldownUntil = currTime + maxTurnDuration + lockoutDuration

            # -------------------------------------------------------------
            # 3. STRAIGHTAWAY WALL AVOIDANCE (ULTRASONIC OR VISION)
            # -------------------------------------------------------------
            if not isTurning:
                if us_hardware_working:
                    # ESP32 handling side ultrasonic wall-centering
                    serial_ctrl.send_command("FORWARD")
                    serial_ctrl.send_command("AUTO_US_ON")
                else:
                    # Disable ESP32 side ultrasonic loop & send camera steering angle
                    serial_ctrl.send_command("AUTO_US_OFF")
                    # leftArea > rightArea => Too close to Left wall => Steer RIGHT (>100)
                    # rightArea > leftArea => Too close to Right wall => Steer LEFT (<100)
                    aDiff = rightArea - leftArea  # Negative when close to left wall
                    steer_angle = int(100 - (aDiff * 0.02))
                    steer_angle = max(60, min(140, steer_angle))
                    serial_ctrl.send_command(f"DRIVE:245:{steer_angle}")

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
            us_mode_str = "US_CENTERING" if us_hardware_working else "VISION_WALLS"
            
            if isTurning:
                t_ela = round(currTime - turnStartTime, 1)
                state_str = f"TURNING ({turnDir.upper()} {t_ela}s)"
            else:
                state_str = f"{us_mode_str} ({turnDir.upper()})"
            
            telemetry_text = f"Cam:{cam_type} | State:{state_str} | Turns:{t}/12 | LineSeen:{lDetected}"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | LineLock:{lock_str}"
            us_text = f"US Sensors -> F:{f_us}cm | L:{l_us}cm | R:{r_us}cm | B:{b_us}cm"

            cv2.putText(img_disp, telemetry_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 204), 2)
            cv2.putText(img_disp, wall_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(img_disp, us_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Update live web stream & snapshot
            streamer.update_frame(img_disp)

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
                "Turn Count": f"{t}/12",
                "Line Lockout": lock_str,
                "Line Detected": lDetected,
                "Wall Steering Mode": us_mode_str,
                "Left Wall Area (px)": leftArea,
                "Right Wall Area (px)": rightArea,
                "US Front (cm)": f_us,
                "US Left (cm)": l_us,
                "US Right (cm)": r_us,
                "US Back (cm)": b_us
            })

            # Stop after 3 full laps (12 turns)
            if t >= 12 and not isTurning:
                print(f"[FINISH] Completed 12 turns (3 laps). Stopping bot!")
                time.sleep(1.0)
                serial_ctrl.send_command("STOP")
                break

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
