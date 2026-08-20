#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 Open Challenge Autonomous Navigation (Round 1)

Universal Camera Support:
- Supports both Pi Camera v2 (Picamera2) AND standard USB Webcams (cv2.VideoCapture).
- Auto-fallbacks to USB Webcam if Picamera2 is unavailable.
- Can be forced to use USB Webcam with '--webcam' or '-w' flag.
"""

import sys
import time
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rOrange, rBlack, rBlue
from wro_functions import find_contours, max_contour, display_roi, display_variables
from camera_streamer import CameraDebugStreamer


class CameraManager:
    """Universal Camera abstraction supporting both Picamera2 and OpenCV USB Webcams."""

    def __init__(self, force_webcam=False, device_index=0):
        self.force_webcam = force_webcam
        self.device_index = device_index
        self.cap = None
        self.picam2 = None
        self.is_webcam = False

    def start(self):
        if self.force_webcam:
            self._start_webcam()
        else:
            try:
                from picamera2 import Picamera2
                print("[INFO] Initializing Picamera2 (Pi CSI Camera)...")
                self.picam2 = Picamera2()
                self.picam2.preview_configuration.main.size = (640, 480)
                self.picam2.preview_configuration.main.format = "RGB888"
                self.picam2.preview_configuration.controls.FrameRate = 30
                self.picam2.preview_configuration.align()
                self.picam2.configure("preview")
                self.picam2.start()
                self.is_webcam = False
                print("[SUCCESS] Picamera2 initialized!")
            except Exception as e:
                print(f"[INFO] Picamera2 not available ({e}). Switching to USB Webcam...")
                self._start_webcam()

    def _start_webcam(self):
        search_indices = [self.device_index, 0, 1, 2, 3, 4, 5, 6, 8]
        seen = set()
        search_indices = [x for x in search_indices if not (x in seen or seen.add(x))]

        for idx in search_indices:
            print(f"[INFO] Testing USB Webcam index {idx}...")
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap and cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        
                        # Test capture 1 frame to verify real webcam device
                        for _ in range(3):
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                print(f"[SUCCESS] USB Webcam initialized on index {idx} (/dev/video{idx})!")
                                self.cap = cap
                                self.device_index = idx
                                self.is_webcam = True
                                return
                        cap.release()
                except Exception:
                    pass

        print("[ERROR] Could not find any working USB webcam across indices 0-8!", file=sys.stderr)
        self.is_webcam = True
        self.cap = None

    def capture_array(self):
        if self.is_webcam:
            if self.cap:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
            return None
        else:
            return self.picam2.capture_array()

    def stop(self):
        if self.is_webcam and self.cap:
            self.cap.release()
        elif self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass


def main():
    print("=" * 65)
    print("   ROBOVANGUARD - WRO Round 1 Open Challenge Node (Pi 5)")
    print("   Hybrid Architecture: ESP32 Side Ultrasonic + Vision Corners")
    print("=" * 65)

    # Check for CLI flags
    force_webcam = "--webcam" in sys.argv or "-w" in sys.argv

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
            img = camera.capture_array()
            if img is None:
                time.sleep(0.01)
                continue

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
                # Continuously stream active turn command to refresh ESP32 500ms watchdog & lock servo angle!
                targetTurnCmd = "TURN_LEFT" if turnDir == "left" else "TURN_RIGHT"
                serial_ctrl.send_command(targetTurnCmd)

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

            cam_type = "WEBCAM" if camera.is_webcam else "PICAM2"
            state_str = f"TURNING ({turnDir.upper()})" if isTurning else f"US_CENTERING ({turnDir.upper()})"
            telemetry_text = f"Cam: {cam_type} | State: {state_str} | Turns: {t}/12"
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
                "Camera Type": cam_type,
                "State": state_str,
                "Track Dir": turnDir,
                "Turn Count": f"{t}/12",
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
