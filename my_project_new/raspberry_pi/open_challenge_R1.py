#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Bench Testing & Monitor Debug Mode:
- Displays live OpenCV window ('WRO Open Challenge (Pi 5)') directly on attached monitor.
- Live Web Camera Debug Streamer (http://<pi_ip>:8080) runs concurrently.
- Press 'q' or 'ESC' in the window or terminal to stop bot immediately.
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
    print("   Bench Testing Mode (Live Monitor Display ON)")
    print("=" * 65)

    # 1. Initialize USB Serial connection to ESP32
    serial_ctrl = WROSerialController()
    if not serial_ctrl.connect():
        print("[ERROR] Cannot proceed without ESP32 serial connection.")
        sys.exit(1)

    # CRITICAL: Force immediate STOP on connect so bot does NOT move during setup!
    print("[SAFETY] Forcing robot STOP state during initialization...")
    serial_ctrl.send_command("STOP")
    time.sleep(0.5)

    # 2. Start Live Web Camera Debug Streamer (port 8080)
    streamer = CameraDebugStreamer(port=8080)
    streamer.start()

    # Enable monitor display window by default
    show_monitor_display = "--no-display" not in sys.argv
    window_name = "WRO Open Challenge - Monitor Debug View (Pi 5)"

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

    # Capture warmup frames to stabilize exposure & show live camera feed on monitor
    print("[INFO] Capturing camera warmup frames...")
    for _ in range(15):
        warmup_frame = picam2.capture_array()
        streamer.update_frame(warmup_frame)
        if show_monitor_display:
            cv2.imshow(window_name, warmup_frame)
            cv2.waitKey(1)
        time.sleep(0.04)

    # 4. Safety Countdown before bot starts driving
    print("\n[READY] Camera & Vision Engine Ready!")
    print("[COUNTDOWN] Bench testing bot starts driving in 3 seconds... (Press 'q' to abort)")
    for c in range(3, 0, -1):
        print(f"[COUNTDOWN] {c}...")
        if show_monitor_display:
            cd_img = warmup_frame.copy()
            cv2.putText(cd_img, f"STARTING IN {c} SECONDS...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow(window_name, cd_img)
            cv2.waitKey(1)
        time.sleep(1.0)

    # 5. Start robot driving forward ONLY AFTER countdown finishes!
    print("[START] Driving FORWARD now!")
    serial_ctrl.send_command("FORWARD")

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    ROI1 = [20, 170, 240, 220]   # Left wall ROI
    ROI2 = [400, 170, 620, 220]  # Right wall ROI
    ROI3 = [200, 300, 440, 350]  # Ground indicator line ROI

    # Navigation flags & state counters
    lTurn = False
    rTurn = False
    t = 0  # Completed turn count

    # Control parameters
    kp = 0.02
    kd = 0.006

    straightConst = 100  # ESP32 servo center (100 deg)
    tDeviation = 20

    # Turn angle thresholds (ESP32 servo range: 40 to 160 deg)
    sharpRight = straightConst + tDeviation  # 120 deg
    sharpLeft = straightConst - tDeviation   # 80 deg
    maxRight = 140
    maxLeft = 60

    turnThresh = 150   # Area below which turn begins
    exitThresh = 1500  # Area above which turn ends

    angle = straightConst
    prevAngle = angle
    aDiff = 0
    prevDiff = 0

    lDetected = False
    turnDir = "none"

    try:
        while True:
            # Capture frame from Pi Camera v2
            img = picam2.capture_array()

            # Convert BGR to LAB color space
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            img_lab = cv2.GaussianBlur(img_lab, (7, 7), 0)

            # Find contours for left/right walls & orange/blue floor lines
            cListLeft = find_contours(img_lab, rBlack, ROI1)
            cListRight = find_contours(img_lab, rBlack, ROI2)
            cListOrange = find_contours(img_lab, rOrange, ROI3)
            cListBlue = find_contours(img_lab, rBlue, ROI3)

            # Calculate wall contour areas
            leftArea = max_contour(cListLeft, ROI1)[0]
            rightArea = max_contour(cListRight, ROI2)[0]

            # Detect orange/blue corner marker lines
            if max_contour(cListOrange, ROI3)[0] > 100:
                lDetected = True
                if turnDir == "none":
                    turnDir = "right"
            elif max_contour(cListBlue, ROI3)[0] > 100:
                lDetected = True
                if turnDir == "none":
                    turnDir = "left"

            # Calculate area difference for wall centering
            aDiff = rightArea - leftArea

            # Compute steering angle using PD control algorithm
            angle = int(straightConst - (aDiff * kp + (aDiff - prevDiff) * kd))
            angle = max(maxLeft, min(maxRight, angle))

            # Turn triggering logic
            if leftArea <= turnThresh and not rTurn:
                lTurn = True
            elif rightArea <= turnThresh and not lTurn:
                rTurn = True

            # In a turn state
            if lTurn or rTurn:
                # Check turn completion condition
                if (rightArea > exitThresh and rTurn) or (leftArea > exitThresh and lTurn):
                    lTurn = False
                    rTurn = False
                    prevDiff = 0
                    if lDetected:
                        t += 1
                        print(f"[NAV EVENT] Completed turn {t}/12")
                        lDetected = False
                elif lTurn:
                    angle = max(angle, sharpLeft)
                elif rTurn:
                    angle = min(angle, sharpRight)
            else:
                # Clamp straight driving angle
                angle = max(sharpLeft, min(sharpRight, angle))

            # Transmit continuous steering angle to ESP32 (refreshes 500ms watchdog)
            serial_ctrl.send_steer(angle)

            # Update tracking variables
            prevDiff = aDiff
            prevAngle = angle

            # Construct telemetry overlay for live monitor display
            img_disp = img.copy()
            img_disp = display_roi(img_disp, [ROI1, ROI2, ROI3])
            cv2.drawContours(img_disp[ROI3[1]:ROI3[3], ROI3[0]:ROI3[2]], cListOrange, -1, (0, 255, 0), 2)
            cv2.drawContours(img_disp[ROI1[1]:ROI1[3], ROI1[0]:ROI1[2]], cListLeft, -1, (0, 255, 0), 2)
            cv2.drawContours(img_disp[ROI2[1]:ROI2[3], ROI2[0]:ROI2[2]], cListRight, -1, (0, 255, 0), 2)

            telemetry_text = f"Steer: {angle} | L_Area: {leftArea} | R_Area: {rightArea} | Turns: {t}/12"
            cv2.putText(img_disp, telemetry_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 204), 2)

            # Update live web stream & snapshot
            streamer.update_frame(img_disp)

            # Display directly on monitor screen
            if show_monitor_display:
                cv2.imshow(window_name, img_disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("[USER INTERRUPT] Stopping bot from monitor GUI...")
                    serial_ctrl.send_command("STOP")
                    break

            display_variables({
                "Left Area": leftArea,
                "Right Area": rightArea,
                "Steer Angle": angle,
                "Turns": t,
                "Marker Detected": lDetected
            })

            # Stop after 3 full laps (12 turns)
            if t >= 12 and abs(angle - straightConst) <= 10:
                print(f"[FINISH] Completed 12 turns (3 laps). Stopping bot!")
                time.sleep(1.0 if turnDir == "left" else 1.5)
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
