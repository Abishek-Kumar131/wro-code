#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Hybrid Sensor-Vision Control Architecture:
- Uses LAB color thresholds from my_old_contour_colorvals_crt.py.
- Line Marker (Orange/Blue) latches lDetected = True and locks line sensing for 5.0 seconds.
- Continuous Wall Drop Detection: Wall area drop (leftArea/rightArea <= turnThresh) triggers corner turn.
- Dynamic Turn Termination: Turning arc STOPS IMMEDIATELY as soon as the wall re-appears
  on the turning side (turnSideWallArea >= wallReappearThresh), returning to Side Ultrasonic Centering.
- ESP32 handles high-frequency Side Ultrasonic Wall Centering (l_us & r_us).
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
    print("   Hybrid Architecture: ESP32 Side Ultrasonic + Pi 5 Vision Corners")
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

    # 5. Start robot driving forward & enable ESP32 side ultrasonic wall-centering!
    print("[START] Driving FORWARD & Enabling ESP32 Side Ultrasonic Centering!")
    serial_ctrl.send_command("FORWARD")
    serial_ctrl.send_command("AUTO_US_ON")

    # Regions of Interest (ROI) [x1, y1, x2, y2] (from my_old_contour_colorvals_crt.py)
    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [200, 300, 440, 350]  # Ground indicator line ROI

    # Navigation flags & state counters
    t = 0                  # Completed turn count (3 laps x 4 turns = 12)
    turnDir = forced_dir   # Track direction ("left", "right", or "none")
    lDetected = False
    isTurning = False
    turnStartTime = 0
    turnCooldownUntil = 0  # Cooldown timer to prevent turn re-triggering
    lineCooldownUntil = 0  # 5.0s lockout timer for floor line marker sensing
    turnThresh = 200       # Area threshold below which wall drop is detected
    wallReappearThresh = 400 # Area threshold above which wall re-appearance STOPS turning!
    maxTurnDuration = 2.5  # Safety max turn duration (seconds)

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

            # -------------------------------------------------------------
            # 1. FLOOR LINE MARKER DETECTION (LATCHES lDetected + 5s LOCKOUT)
            # -------------------------------------------------------------
            if not isTurning and currTime >= lineCooldownUntil:
                if orangeArea > 150 and orangeArea > blueArea:
                    lDetected = True
                    if forced_dir == "none":
                        turnDir = "right"
                    lineCooldownUntil = currTime + 5.0  # 5.0s lockout for floor line marker sensing!
                    print(f"[VISION MARKER] Detected ORANGE Line ({orangeArea} px) -> Track Dir = RIGHT (CW). 5s Line Lockout Active.")
                elif blueArea > 150 and blueArea > orangeArea:
                    lDetected = True
                    if forced_dir == "none":
                        turnDir = "left"
                    lineCooldownUntil = currTime + 5.0  # 5.0s lockout for floor line marker sensing!
                    print(f"[VISION MARKER] Detected BLUE Line ({blueArea} px) -> Track Dir = LEFT (CCW). 5s Line Lockout Active.")

            # -------------------------------------------------------------
            # 2. HYBRID CORNER TURN EXECUTION & DYNAMIC WALL RE-APPEARANCE EXIT
            # -------------------------------------------------------------
            if isTurning:
                # Continuously stream active turn command to refresh ESP32 500ms watchdog & lock servo angle!
                targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                serial_ctrl.send_command(targetTurnCmd)

                # Check if the wall has RE-APPEARED on the turning side!
                turnSideWallArea = leftArea if turnDir == "left" else rightArea
                wallReappeared = (turnSideWallArea >= wallReappearThresh) and (currTime - turnStartTime >= 0.5)
                turnTimedOut = (currTime - turnStartTime) >= maxTurnDuration

                # STOP TURNING as soon as wall re-appears OR safety timeout occurs!
                if wallReappeared or turnTimedOut:
                    isTurning = False
                    turnCooldownUntil = currTime + 3.0  # 3.0s cooldown before next corner turn can trigger
                    serial_ctrl.send_command("FORWARD")
                    serial_ctrl.send_command("AUTO_US_ON")
                    reason_str = f"Wall Re-appeared ({turnSideWallArea} px)" if wallReappeared else "Max Duration Timeout"
                    print(f"[NAV EVENT] Stop Turn {t}/12 ({turnDir.upper()}) -> {reason_str}! Re-enabled Side Ultrasonic Centering.")

            elif currTime >= turnCooldownUntil:
                # Continuous Wall Drop Check (wall area drops below turnThresh on turn side)
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

                # TRIGGER CORNER TURN: Requires line marker detection (or forced dir) AND wall drop!
                if (lDetected or forced_dir != "none") and wallDropDetected:
                    targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                    t += 1
                    print(f"[NAV EVENT] Marker Seen + Wall Drop Detected! (L:{leftArea} R:{rightArea}) -> Triggering {targetTurnCmd} ({t}/12)...")
                    serial_ctrl.send_command(targetTurnCmd)
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False  # Reset marker flag for next straightaway!
                    lineCooldownUntil = currTime + 5.0  # Lockout floor line sensing for 5.0s from turn start
                    turnCooldownUntil = currTime + 5.0  # Lockout turn re-triggers for 5.0s

            # Keep 500ms serial watchdog refreshed when driving straight
            if not isTurning:
                serial_ctrl.send_command("FORWARD")

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
            line_lock_rem = max(0.0, round(lineCooldownUntil - currTime, 1))
            line_lock_str = f"LOCKED ({line_lock_rem}s)" if line_lock_rem > 0 else "READY"
            state_str = f"TURNING ({turnDir.upper()})" if isTurning else f"US_CENTERING ({turnDir.upper()})"
            telemetry_text = f"Cam:{cam_type} | State:{state_str} | Turns:{t}/12 | LineSense:{line_lock_str}"
            wall_text = f"Walls -> Left:{leftArea}px | Right:{rightArea}px | LineSeen:{lDetected}"
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
                "Line Sensing": line_lock_str,
                "Line Marker Seen": lDetected,
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
