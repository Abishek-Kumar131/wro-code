#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Hybrid Sensor-Vision Control Architecture:
- 3.5-Second Line & Turn Lockout after any line detection or turn trigger.
- Dual Wall Avoidance: Uses ESP32 Side Ultrasonic sensors when connected;
  falls back to Pi 5 Camera Black Wall Contours when US sensors are offline.
- Dynamically steers AWAY from walls in all driving modes.
"""

import sys
import time
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rOrange, rBlack, rBlue
from wro_functions import (CameraManager, find_contours, max_contour, draw_roi,
                           draw_offset_contours, display_variables)
from camera_streamer import CameraDebugStreamer


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 1 Open Challenge Node (Pi 5)")
    print("   Hybrid Architecture: Dual Ultrasonic + Vision Wall Avoidance")
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
    print("[START] Driving FORWARD with Wall Avoidance!")
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
    turnCooldownUntil = 0  # Cooldown timer to lock out lines and turns
    lockoutDuration = 3.5  # Exactly 3.5 seconds lockout after line/turn trigger
    turnDuration = 2.0     # 2.0 seconds corner arc turn duration
    turnThresh = 200       # Area threshold below which wall end is detected

    try:
        while True:
            img = camera.capture_array()
            if img is None:
                time.sleep(0.01)
                continue

            currTime = time.time()
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

            # Find contours using exact LAB color thresholds
            cListLeft = find_contours(img_lab, rBlack, ROI1)
            cListRight = find_contours(img_lab, rBlack, ROI2)
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
            us_online = (l_us > 0 or r_us > 0)

            # -------------------------------------------------------------
            # LINE MARKER DETECTION (WITH 3.5-SECOND LOCKOUT)
            # -------------------------------------------------------------
            if not isTurning and currTime >= turnCooldownUntil:
                if orangeArea > 150 and orangeArea > blueArea:
                    lDetected = True
                    if forced_dir == "none":
                        turnDir = "right"
                    turnCooldownUntil = currTime + lockoutDuration  # 3.5s lockout
                    print(f"[VISION MARKER] Detected ORANGE Line ({orangeArea} px) -> Track Dir = RIGHT (3.5s Lockout Active)")
                elif blueArea > 150 and blueArea > orangeArea:
                    lDetected = True
                    if forced_dir == "none":
                        turnDir = "left"
                    turnCooldownUntil = currTime + lockoutDuration  # 3.5s lockout
                    print(f"[VISION MARKER] Detected BLUE Line ({blueArea} px) -> Track Dir = LEFT (3.5s Lockout Active)")

            # -------------------------------------------------------------
            # HYBRID CORNER TURN TRIGGERING (STRICT LINE + WALL DROP)
            # -------------------------------------------------------------
            if isTurning:
                # Continuously stream active turn command to refresh ESP32 500ms watchdog & lock servo angle!
                targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                serial_ctrl.send_command(targetTurnCmd)

                if currTime - turnStartTime >= turnDuration:
                    isTurning = False
                    turnCooldownUntil = currTime + lockoutDuration  # 3.5s cooldown after turn completes
                    if us_online:
                        serial_ctrl.send_command("FORWARD")
                        serial_ctrl.send_command("AUTO_US_ON")
                    print(f"[NAV EVENT] Completed turn {t}/12 ({turnDir.upper()}). Resumed Wall Avoidance!")
            elif currTime >= turnCooldownUntil:
                # Wall drop check (wall area drops below turnThresh)
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

                # STRICT TRIGGER: Require line marker detection AND wall drop!
                if (lDetected or forced_dir != "none") and wallDropDetected:
                    targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                    t += 1
                    print(f"[NAV EVENT] Marker Seen + Wall Drop! (L:{leftArea} R:{rightArea}) -> Triggering {targetTurnCmd} ({t}/12)...")
                    serial_ctrl.send_command(targetTurnCmd)
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False  # Reset marker flag for next straightaway!
                    turnCooldownUntil = currTime + (turnDuration + lockoutDuration)  # Lockout for turn + 3.5s

            # -------------------------------------------------------------
            # STRAIGHTAWAY WALL AVOIDANCE (ULTRASONIC OR CAMERA FALLBACK)
            # -------------------------------------------------------------
            if not isTurning:
                if us_online:
                    # ESP32 handling side ultrasonic wall-centering
                    serial_ctrl.send_command("FORWARD")
                    serial_ctrl.send_command("AUTO_US_ON")
                else:
                    # Pi 5 Vision Wall Centering (Steer AWAY from walls)
                    # If left wall is bigger (closer), steer RIGHT (>100)
                    # If right wall is bigger (closer), steer LEFT (<100)
                    aDiff = rightArea - leftArea  # Negative when close to left wall
                    steer_angle = int(100 - (aDiff * 0.015))
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
            cd_rem = max(0.0, round(turnCooldownUntil - currTime, 1))
            cd_str = f"LOCKED ({cd_rem}s)" if cd_rem > 0 else "READY"
            us_str = "US_ONLINE" if us_online else "CAM_FALLBACK"
            state_str = f"TURNING ({turnDir.upper()})" if isTurning else f"{us_str} ({turnDir.upper()})"
            telemetry_text = f"Cam:{cam_type} | State:{state_str} | Turns:{t}/12 | LineSeen:{lDetected}"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | Lockout:{cd_str}"
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
                "Lockout (3.5s)": cd_str,
                "Wall Avoidance": us_str,
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
