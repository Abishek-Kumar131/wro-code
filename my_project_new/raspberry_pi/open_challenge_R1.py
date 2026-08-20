#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Hybrid Sensor-Vision Control Architecture:
- ESP32 handles high-frequency Side Ultrasonic Wall Centering (l_us & r_us).
- Pi 5 Camera (Picamera2) detects Floor Direction Markers (Blue/Orange) & Wall Ends.
- Requires both Line Detection (lDetected == True) AND Wall Drop before triggering corner turn.
- Displays live Ultrasonic Sensor Readings (F, L, R, B) in terminal and OpenCV display.
"""

import sys
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from wro_serial import WROSerialController
from masks import rOrange, rBlack, rBlue
from wro_functions import find_contours, max_contour, display_roi, display_variables
from camera_streamer import CameraDebugStreamer


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 1 Open Challenge Node (Pi 5)")
    print("   Hybrid Architecture: ESP32 Side Ultrasonic + Pi 5 Vision Corners")
    print("=" * 65)

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

    # 3. Initialize Pi Camera v2 via Picamera2
    print("[INFO] Initializing Picamera2...")
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.controls.FrameRate = 30
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()
    print("[SUCCESS] Picamera2 started!")

    # Warmup camera frames
    print("[INFO] Capturing camera warmup frames...")
    for _ in range(15):
        warmup_frame = picam2.capture_array()
        streamer.update_frame(warmup_frame)
        if show_monitor_display:
            cv2.imshow(window_name, warmup_frame)
            cv2.waitKey(1)
        time.sleep(0.04)

    # 4. Safety Countdown before bot starts driving
    print("\n[READY] Hybrid Sensor-Vision Engine Ready!")
    print("[COUNTDOWN] Bot starts driving in 3 seconds... (Press 'q' to abort)")
    for c in range(3, 0, -1):
        print(f"[COUNTDOWN] {c}...")
        if show_monitor_display:
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

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    ROI1 = [10, 150, 260, 240]   # Left wall end ROI
    ROI2 = [380, 150, 630, 240]  # Right wall end ROI
    ROI3 = [200, 300, 440, 360]  # Ground indicator line ROI

    # Navigation flags & state counters
    t = 0                  # Completed turn count (3 laps x 4 turns = 12)
    turnDir = "none"       # Fixed turn direction ("left" or "right") once first floor line seen
    lDetected = False
    isTurning = False
    turnStartTime = 0
    turnCooldownUntil = 0
    turnDuration = 2.0     # 2.0 seconds corner arc turn duration
    turnThresh = 150       # Area threshold below which wall end is detected

    try:
        while True:
            img = picam2.capture_array()
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img_lab = cv2.GaussianBlur(img_lab, (7, 7), 0)

            cListLeft = find_contours(img_lab, rBlack, ROI1)
            cListRight = find_contours(img_lab, rBlack, ROI2)
            cListOrange = find_contours(img_lab, rOrange, ROI3)
            cListBlue = find_contours(img_lab, rBlue, ROI3)

            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]

            # Get latest ultrasonic sensor telemetry from ESP32
            us_data = serial_ctrl.get_us_data()
            f_us = us_data.get("f", 0)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

            # Detect floor markers (Orange = Clockwise / Right Turn, Blue = Counter-Clockwise / Left Turn)
            if max_contour(cListOrange, ROI3)[0] > 100:
                lDetected = True
                if turnDir == "none":
                    turnDir = "right"
                    print("[VISION MARKER] Detected ORANGE Line -> Set Track Direction = RIGHT (CW)")
            elif max_contour(cListBlue, ROI3)[0] > 100:
                lDetected = True
                if turnDir == "none":
                    turnDir = "left"
                    print("[VISION MARKER] Detected BLUE Line -> Set Track Direction = LEFT (CCW)")

            currTime = time.time()

            # -------------------------------------------------------------
            # HYBRID CORNER TURN TRIGGERING (STRICT LINE + WALL DROP)
            # -------------------------------------------------------------
            if isTurning:
                # Check if 2.0s turn duration has elapsed
                if currTime - turnStartTime >= turnDuration:
                    isTurning = False
                    turnCooldownUntil = currTime + 1.2  # 1.2s cooldown before next turn
                    serial_ctrl.send_command("FORWARD")
                    serial_ctrl.send_command("AUTO_US_ON")
                    print(f"[NAV EVENT] Completed turn {t}/12. Re-enabled Side Ultrasonic Centering!")
            elif currTime >= turnCooldownUntil:
                # Wall drop check
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

                # STRICT TRIGGER: Require line marker detection AND wall drop!
                if lDetected and wallDropDetected:
                    targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                    t += 1
                    print(f"[NAV EVENT] Marker + Wall End Detected! Triggering {targetTurnCmd} (Turn {t}/12)...")
                    serial_ctrl.send_command(targetTurnCmd)
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False  # Reset marker flag until next line is seen!

            # Keep 500ms serial watchdog refreshed when driving straight
            if not isTurning:
                serial_ctrl.send_command("FORWARD")

            # Telemetry display overlay
            img_disp = img.copy()
            img_disp = display_roi(img_disp, [ROI1, ROI2, ROI3])
            cv2.drawContours(img_disp[ROI3[1]:ROI3[3], ROI3[0]:ROI3[2]], cListOrange, -1, (0, 255, 0), 2)
            cv2.drawContours(img_disp[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]], cListLeft, -1, (0, 255, 0), 2)
            cv2.drawContours(img_disp[ROI2[1]:ROI2[3], ROI2[0]:ROI2[2]], cListRight, -1, (0, 255, 0), 2)

            state_str = f"TURNING ({turnDir.upper()})" if isTurning else f"US_CENTERING ({turnDir.upper()})"
            telemetry_text = f"State: {state_str} | Turns: {t}/12 | LineSeen: {lDetected}"
            us_text = f"US Sensors -> F:{f_us}cm | L:{l_us}cm | R:{r_us}cm | B:{b_us}cm"

            cv2.putText(img_disp, telemetry_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 204), 2)
            cv2.putText(img_disp, us_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            # Update live web stream & snapshot
            streamer.update_frame(img_disp)

            # Display directly on monitor screen
            if show_monitor_display:
                cv2.imshow(window_name, img_disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("[USER INTERRUPT] Stopping bot from monitor GUI...")
                    serial_ctrl.send_command("STOP")
                    break

            display_variables({
                "State": state_str,
                "Track Dir": turnDir,
                "Turn Count": f"{t}/12",
                "Line Detected": lDetected,
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
        time.sleep(0.1)
        serial_ctrl.disconnect()
        if show_monitor_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
