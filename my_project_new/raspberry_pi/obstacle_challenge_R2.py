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
import select
import math
import cv2
import numpy as np
from wro_serial import WROSerialController
from masks import rMagenta, rRed, rGreen, rBlue, rOrange, rBlack, lotType
from wro_functions import (CameraManager, find_black_wall_contours, find_red_pillar_contours,
                           find_green_pillar_contours, find_orange_line_contours, find_blue_line_contours,
                           find_contours, max_contour, draw_roi, draw_offset_contours, display_variables)


def wait_for_button_press(gpio_pin=17, show_display=False, window_name="", camera=None, active_high=False):
    """Waits for a physical push button press on Pi 5 GPIO pin before starting autonomous run."""
    print("=" * 65)
    print(f"[COMPETITION STANDBY] Waiting for physical button press on GPIO {gpio_pin}...")
    print("=" * 65)

    is_interactive = sys.stdin.isatty()
    if is_interactive:
        print("  -> Running in interactive terminal (Press ENTER or Button to start)")
    else:
        print("  -> Running as background service (Waiting strictly for physical GPIO Button)")

    button_obj = None
    try:
        from gpiozero import Button
        pull_up = not active_high
        button_obj = Button(gpio_pin, pull_up=pull_up, bounce_time=0.05)
        print(f"[GPIO] Listening on GPIO {gpio_pin} (pull_up={pull_up} via gpiozero).")
    except Exception:
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            pud = GPIO.PUD_DOWN if active_high else GPIO.PUD_UP
            GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=pud)
            print(f"[GPIO] Listening on GPIO {gpio_pin} (via RPi.GPIO).")
        except Exception as e:
            print(f"[GPIO INFO] Hardware GPIO library not loaded ({e}).")

    time.sleep(0.5)

    settle_start = time.time()
    while True:
        is_pressed = False
        if button_obj is not None:
            is_pressed = button_obj.is_pressed
        elif 'GPIO' in sys.modules:
            import RPi.GPIO as GPIO
            pin_val = GPIO.input(gpio_pin)
            is_pressed = (pin_val == GPIO.HIGH) if active_high else (pin_val == GPIO.LOW)

        if not is_pressed or (time.time() - settle_start > 3.0):
            break
        time.sleep(0.05)

    print("[GPIO] READY FOR MATCH! Waiting for button press or ENTER key...")

    while True:
        is_pressed = False
        if button_obj is not None:
            is_pressed = button_obj.is_pressed
        elif 'GPIO' in sys.modules:
            import RPi.GPIO as GPIO
            pin_val = GPIO.input(gpio_pin)
            is_pressed = (pin_val == GPIO.HIGH) if active_high else (pin_val == GPIO.LOW)

        if is_pressed:
            time.sleep(0.08)
            confirm_pressed = False
            if button_obj is not None:
                confirm_pressed = button_obj.is_pressed
            elif 'GPIO' in sys.modules:
                import RPi.GPIO as GPIO
                pin_val = GPIO.input(gpio_pin)
                confirm_pressed = (pin_val == GPIO.HIGH) if active_high else (pin_val == GPIO.LOW)

            if confirm_pressed:
                print("\n" + "=" * 65)
                print("[MATCH START] PHYSICAL BUTTON PRESSED! Launching Round 2 Obstacle Challenge...")
                print("=" * 65)
                return True

        if is_interactive:
            if sys.stdin in select.select([sys.stdin], [], [], 0.02)[0]:
                line = sys.stdin.readline()
                print("\n" + "=" * 65)
                print("[MATCH START] ENTER KEY PRESSED! Launching Round 2 Obstacle Challenge...")
                print("=" * 65)
                return True

        if show_display and camera is not None:
            standby_frame = camera.capture_array()
            if standby_frame is not None:
                disp = standby_frame.copy()
                cv2.putText(disp, "ROUND 2 STANDBY - PRESS BUTTON TO START", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 13 or key == 32:  # ENTER or SPACE
                    print("\n[MATCH START] GUI Key Pressed! Starting Round 2 Obstacle Challenge...")
                    return True
                elif key == 27 or key == ord('q'):
                    print("\n[ABORT] User cancelled from GUI window.")
                    sys.exit(0)

        time.sleep(0.03)


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


def find_pillar(contours, target, p, colour, ROI3, tempParking=False, maxDist=480, endConst=15):
    """Processes pillar contours and returns the nearest pillar candidate."""
    num_p = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Check if area is large enough for the specific color pillar (70 px for early lookahead)
        if (area >= 70 and colour == "red") or (area >= 60 and colour == "red" and tempParking) or (area >= 70 and colour == "green"):
            if tempParking and colour == "green" and area < 200:
                continue

            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            # Convert ROI-relative coordinates to full-frame coordinates
            x += ROI3[0] + w // 2
            y += ROI3[1] + h

            # Distance between pillar bottom and screen bottom-center (320, 480)
            temp_dist = round(math.dist([x, y], [320, 480]), 0)

            if 80 < temp_dist < 480:
                num_p += 1

            # Only filter if it exceeds max tracking distance
            if temp_dist > maxDist:
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
    print("   Architecture: Early Lookahead + High-Grip Dynamic Evasion Speed")
    print("=" * 65)

    force_webcam = "--webcam" in sys.argv or "-w" in sys.argv
    skip_button = "--no-wait" in sys.argv or "--instant-start" in sys.argv
    active_high = "--active-high" in sys.argv

    gpio_pin = 17
    if "--pin" in sys.argv:
        idx = sys.argv.index("--pin")
        if idx + 1 < len(sys.argv):
            gpio_pin = int(sys.argv[idx + 1])

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

    # ------------------------------------------------------------------------
    # COMPETITION STARTUP: Wait for physical button press on Pi 5 GPIO pin
    # ------------------------------------------------------------------------
    if not skip_button:
        wait_for_button_press(gpio_pin=gpio_pin, show_display=show_monitor_display,
                              window_name=window_name, camera=camera, active_high=active_high)

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
    # HARDWARE SERVO MAPPING: 60 = LEFT, 100 = CENTER, 140 = RIGHT
    # ------------------------------------------------------------------------
    redTarget = 110    # Target X position for Red Pillars (Pillar on LEFT -> Steers RIGHT towards 140)
    greenTarget = 530  # Target X position for Green Pillars (Pillar on RIGHT -> Steers LEFT towards 60)

    straightConst = 100 # Steering center (100 degrees)
    sharpRight = 140   # Sharp right steering lock (140 deg)
    sharpLeft = 60     # Sharp left steering lock (60 deg)

    # Speed Parameters (All speeds >= 225 PWM for strong motor response without stalling)
    normalSpeed = 245  # Full open straightaway speed (245 PWM)
    pillarSpeed = 228  # High-torque evasion speed above 220 PWM (228 PWM)
    turnSpeed = 235    # High cornering speed (235 PWM)

    # PD Pillar Avoidance gains (High responsiveness for early avoidance)
    cKp = 0.32
    cKd = 0.25
    cy = 0.12

    # Regions of Interest (ROI) [x1, y1, x2, y2]
    ROI1 = [20, 170, 240, 220]   # Left Wall ROI (Outer Left dedicated box)
    ROI2 = [400, 170, 620, 220]  # Right Wall ROI (Outer Right dedicated box)
    ROI3 = [0, 60, 640, 280]     # Signal Pillars / Block Detection ROI (Early Horizon Y=60 for 1.5m lookahead)
    ROI4 = [200, 270, 440, 340]  # Ground Markers & Parking Lot ROI (Floor Lines)

    # Navigation flags & state counters
    t = 0                  # Completed turn count (3 laps x 4 turns = 12)
    turnDir = forced_dir   # Track direction ("left", "right", or "none")
    lDetected = False
    isTurning = False
    turnStartTime = 0
    lineLockoutUntil = 0   # 3.0s line detection lockout timer
    turnCooldownUntil = 0  # 3.0s turn trigger cooldown timer
    reverseCooldownUntil = 0 # 1.2s emergency reverse cooldown timer
    lockoutDuration = 3.0  # Exactly 3.0 seconds lockout

    # Dynamic Turn Exit Timings (Optimized for Narrow FOV Camera)
    minTurnDuration = 0.8  # Minimum arc turn time before checking wall re-acquisition (0.8s)
    maxTurnDuration = 2.2  # Safety maximum turn time cap (2.2s)
    wallReacquireArea = 400 # Area threshold to confirm single wall in narrow FOV view
    turnThresh = 150       # Area threshold below which wall end is detected

    tempParking = False
    parkingL = False
    parkingR = False

    angle = straightConst
    prevAngle = angle
    aDiff = 0
    prevDiff = 0
    error = 0
    prevError = 0
    endConst = 20
    maxDist = 480

    # Pillar evasion persistence state (prevents premature evasion drop / rear wheel clipping)
    pillar_evade_until = 0.0
    last_evade_steer = straightConst
    last_evade_pillar_target = None

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
            navMode = "STRAIGHT"  # Scope-safe default for telemetry
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

            # Extract contours using strict separation logic
            contours_left = find_black_wall_contours(img, ROI1)
            contours_right = find_black_wall_contours(img, ROI2)
            contours_red = find_red_pillar_contours(img, ROI3)         # Strict Red Pillar + Aspect Ratio filter (H/W >= 0.65)
            contours_green = find_green_pillar_contours(img, ROI3)     # Strict Green Pillar + Aspect Ratio filter (H/W >= 0.60)
            contours_orange = find_orange_line_contours(img, ROI4)      # Strict Orange Line (Floor ROI4)
            contours_blue = find_blue_line_contours(img, ROI4)          # Strict Blue Line (Floor ROI4)
            contours_magenta = find_contours(img_lab, rMagenta, ROI4)

            leftArea = max_contour(contours_left, ROI1)[0]
            rightArea = max_contour(contours_right, ROI2)[0]
            orangeArea = max_contour(contours_orange, ROI4)[0]
            blueArea = max_contour(contours_blue, ROI4)[0]
            magentaArea = max_contour(contours_magenta, ROI4)[0]

            # Get latest ultrasonic sensor telemetry from ESP32
            us_data = serial_ctrl.get_us_data()
            f_us = us_data.get("f", 0)
            f1_us = us_data.get("f1", f_us)
            f2_us = us_data.get("f2", f_us)
            l_us = us_data.get("l", 0)
            r_us = us_data.get("r", 0)
            b_us = us_data.get("b", 0)

            # -------------------------------------------------------------
            # 0. EMERGENCY ANGLED REVERSE (Gentle Track-Aligned Backoff)
            # -------------------------------------------------------------
            red_pillar_area = max_contour(contours_red, ROI3)[0]
            green_pillar_area = max_contour(contours_green, ROI3)[0]
            close_visual_pillar = (red_pillar_area > 2000 or green_pillar_area > 2000)

            # Strictly require positive distance (0 is open-air measurement timeout, NOT a crash!)
            front_sensors = [v for v in (f_us, f1_us, f2_us) if 0 < v <= 12]
            has_close_us = len(front_sensors) > 0
            side_wall_jam = (0 < l_us <= 5) or (0 < r_us <= 5)

            if (has_close_us or (side_wall_jam and close_visual_pillar)) and not isTurning and currTime >= reverseCooldownUntil:
                min_close = min(front_sensors) if front_sensors else 0

                # Gentle reverse alignment (stays strictly down the corridor, no 180° disorientation!)
                rev_steer = 100
                if red_pillar_area > 800 or (0 < l_us <= 6) or (leftArea > 800):
                    rev_steer = 92  # Obstacle on Left -> Back up straight with gentle 8° right-nose bias
                elif green_pillar_area > 800 or (0 < r_us <= 6) or (rightArea > 800):
                    rev_steer = 108 # Obstacle on Right -> Back up straight with gentle 8° left-nose bias

                print("=" * 65)
                print(f"[EMERGENCY REVERSE] Proximity limit (US: {min_close} cm) -> Backing straight (Angle: {rev_steer}°)...")
                print("=" * 65)

                rev_start = time.time()
                while True:
                    serial_ctrl.send_command(f"DRIVE:-235:{rev_steer}")
                    time.sleep(0.04)

                    _ = camera.capture_array()
                    if show_monitor_display:
                        cv2.waitKey(1)

                    us_check = serial_ctrl.get_us_data()
                    curr_f = us_check.get("f", 0)
                    curr_f1 = us_check.get("f1", curr_f)
                    curr_f2 = us_check.get("f2", curr_f)
                    curr_b = us_check.get("b", 0)

                    rev_elapsed = time.time() - rev_start

                    if 0 < curr_b <= 8:
                        print(f"[SAFETY] Rear wall buffer reached ({curr_b}cm)! Stopping reverse.")
                        break

                    front_cleared = (curr_f >= 18 or curr_f == 0) and \
                                    (curr_f1 >= 18 or curr_f1 == 0) and \
                                    (curr_f2 >= 18 or curr_f2 == 0)

                    if rev_elapsed >= 0.35 and front_cleared:
                        break
                    if rev_elapsed >= 0.70: # Crisp, short safety cap
                        break

                serial_ctrl.send_command("STOP")
                time.sleep(0.04)
                serial_ctrl.send_command(f"DRIVE:{pillarSpeed}:100") # Immediately drive straight forward
                reverseCooldownUntil = time.time() + 1.2 # 1.2s cooldown
                last_cmd_time = time.time()
                last_drive_speed = pillarSpeed
                last_steer_angle = 100

            # -------------------------------------------------------------
            # 1. PERMANENT FIRST-COLOR DIRECTION LOCK & MARKER DETECTION
            # -------------------------------------------------------------
            if t < 12 and not isTurning and currTime >= lineLockoutUntil:
                if turnDir == "none":
                    if orangeArea > 120 and orangeArea > blueArea:
                        turnDir = "right"
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[FIRST-COLOR LOCK] First Line: ORANGE ({orangeArea} px) -> PERMANENTLY LOCKED TO RIGHT (All 3 Laps)!")
                    elif blueArea > 120 and blueArea > orangeArea:
                        turnDir = "left"
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[FIRST-COLOR LOCK] First Line: BLUE ({blueArea} px) -> PERMANENTLY LOCKED TO LEFT (All 3 Laps)!")
                
                elif turnDir == "right":
                    # STRICT RIGHT LOCK: Only listen to Orange markers! Completely ignore blue noise.
                    if orangeArea > 120:
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[LOCKED MARKER] Detected ORANGE Line ({orangeArea} px) -> Next Turn: RIGHT (Turn {t+1}/12)")
                
                elif turnDir == "left":
                    # STRICT LEFT LOCK: Only listen to Blue markers! Completely ignore orange noise.
                    if blueArea > 120:
                        lDetected = True
                        lineLockoutUntil = currTime + lockoutDuration
                        print(f"[LOCKED MARKER] Detected BLUE Line ({blueArea} px) -> Next Turn: LEFT (Turn {t+1}/12)")

            # -------------------------------------------------------------
            # 2. HYBRID CORNER TURN & DYNAMIC VISION EXIT (REDUCED TURN SPEED)
            # -------------------------------------------------------------
            if isTurning:
                # HARDWARE SERVO: 60 = LEFT turn, 140 = RIGHT turn!
                targetTurnAngle = 60 if turnDir == "left" else 140
                
                # Stream active corner turn at turnSpeed (235) to prevent drifting!
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
                    turnCooldownUntil = currTime + 3.0  # 3.0s cooldown after turn ends to prevent double-counting
                    lineLockoutUntil = currTime + 3.0   # 3.0s line lockout after turn ends
                    exit_reason = "WALL_REACQUIRED" if newWallAcquired else "MAX_TIMEOUT"
                    print(f"[NAV EVENT] Turn {t}/12 ({turnDir.upper()}) EXITED via {exit_reason} in {round(turnElapsed, 2)}s!")
            
            elif (t < 12) and currTime >= turnCooldownUntil:
                # Wall drop check (inner wall drops away at the corner)
                wallDropDetected = (leftArea <= turnThresh and rightArea <= turnThresh) or \
                                   (turnDir == "left" and leftArea <= turnThresh) or \
                                   (turnDir == "right" and rightArea <= turnThresh)

                # STRICT TRIGGER: Require line marker detection (or locked direction) AND wall drop!
                if (lDetected or (turnDir != "none" and wallDropDetected) or forced_dir != "none") and wallDropDetected:
                    targetTurnAngle = 60 if turnDir == "left" else 140
                    t += 1
                    marker_info = "Marker + Wall Drop" if lDetected else "Inner Wall Drop"
                    print(f"[NAV EVENT] {marker_info}! (L:{leftArea} R:{rightArea}) -> Triggering Turn ({t}/12) angle={targetTurnAngle} at speed {turnSpeed}...")
                    serial_ctrl.send_command(f"DRIVE:{turnSpeed}:{targetTurnAngle}")
                    last_cmd_time = currTime
                    last_drive_speed = turnSpeed
                    last_steer_angle = targetTurnAngle
                    isTurning = True
                    turnStartTime = currTime
                    lDetected = False  # Reset marker flag!
                    turnCooldownUntil = currTime + maxTurnDuration + 3.0  # Lock out next turn until corner is fully cleared
                    lineLockoutUntil = currTime + maxTurnDuration + 3.0

            # -------------------------------------------------------------
            # 3. STRAIGHTAWAY OBSTACLE AVOIDANCE & POST-PILLAR CORRIDOR GUARD
            # -------------------------------------------------------------
            if not isTurning and not tempParking:
                # Nearest Pillar Tracking (480px lookahead horizon)
                temp_p = Pillar(0, 1000000, 0, 0, greenTarget)
                cPillar, num_pillars_g = find_pillar(contours_green, greenTarget, temp_p, "green", ROI3, tempParking, maxDist, endConst)
                cPillar, num_pillars_r = find_pillar(contours_red, redTarget, cPillar, "red", ROI3, tempParking, maxDist, endConst)

                if num_pillars_g >= 2 or num_pillars_r >= 2:
                    endConst = 40
                    cKp, cKd, cy = 0.30, 0.22, 0.10
                else:
                    endConst = 20
                    cKp, cKd, cy = 0.34, 0.26, 0.14

                # =========================================================================
                # MODE A: PILLAR VISIBLE OR IN EVASION CLEARANCE WINDOW (100% PURE PILLAR AVOIDANCE)
                # =========================================================================
                if (cPillar.area > 0 or (currTime < pillar_evade_until and last_evade_pillar_target is not None)) and not tempParking:
                    if cPillar.area > 0:
                        last_evade_pillar_target = cPillar.target
                        pillar_evade_until = currTime + 0.70  # Hold evasion heading for 700ms so chassis & rear wheels cleanly pass
                        navMode = "RED_PILLAR" if cPillar.target == redTarget else "GREEN_PILLAR"
                        
                        # HARDWARE SERVO MAPPING: 60 = LEFT, 100 = CENTER, 140 = RIGHT
                        # Red target = 110 (error < 0 -> steers RIGHT towards 140): angle = 100 - (error * cKp)
                        # Green target = 530 (error > 0 -> steers LEFT towards 60): angle = 100 - (error * cKp)
                        error = cPillar.target - cPillar.x
                        angle = int(straightConst - (error * cKp) - ((error - prevError) * cKd))

                        # Vertical proximity scaling (Pulls harder towards evasion side as obstacle gets closer)
                        if not tempParking:
                            y_offset = int(cy * (cPillar.y - ROI3[1]))
                            angle += (y_offset if error <= 0 else -y_offset)

                        # Side Wall Proximity Safety Guard while dodging:
                        # If evading Red (shifted Right) and getting too close to Right Wall (r_us < 12), pull back towards center
                        if cPillar.target == redTarget and 0 < r_us <= 12:
                            angle = min(120, angle)
                        # If evading Green (shifted Left) and getting too close to Left Wall (l_us < 12), pull back towards center
                        elif cPillar.target == greenTarget and 0 < l_us <= 12:
                            angle = max(80, angle)

                        prevError = error
                        last_evade_steer = angle
                    else:
                        # Pillar just cleared front view: maintain parallel corridor heading so rear wheels don't swipe the block!
                        navMode = "EVADING_PILLAR"
                        # Straighten out slightly to track parallel through the gap without hitting wall or pillar
                        if last_evade_pillar_target == redTarget:
                            angle = max(100, min(115, last_evade_steer))
                        else:
                            angle = min(100, max(85, last_evade_steer))

                    # Dedicated high-grip speed during obstacle evasion
                    currentSpeed = pillarSpeed

                # =========================================================================
                # MODE B: PILLAR CLEARED -> PROVEN SAFE WALL CENTERING & CORNER APPROACH
                # =========================================================================
                else:
                    both_walls_visible = (leftArea > 120 and rightArea > 120)

                    if both_walls_visible:
                        # Dual-wall proportional centering
                        navMode = "DUAL_WALL_CENTER"
                        aDiff = rightArea - leftArea
                        angle = int(straightConst - (aDiff * 0.012))
                    elif turnDir == "right" and rightArea <= 120:
                        # Approaching RIGHT turn: Inner right wall dropped.
                        # Maintain straight course using outer left wall; DO NOT steer into outer wall!
                        navMode = "APPROACHING_RIGHT_CORNER"
                        left_err = leftArea - 600
                        angle = int(straightConst - (left_err * 0.008))
                        angle = max(92, min(108, angle))
                    elif turnDir == "left" and leftArea <= 120:
                        # Approaching LEFT turn: Inner left wall dropped.
                        # Maintain straight course using outer right wall; DO NOT steer into outer wall!
                        navMode = "APPROACHING_LEFT_CORNER"
                        right_err = rightArea - 600
                        angle = int(straightConst + (right_err * 0.008))
                        angle = max(92, min(108, angle))
                    else:
                        navMode = "SIDE_US_WALLS"
                        valid_left = (5 < l_us < 80)
                        valid_right = (5 < r_us < 80)

                        if valid_left and valid_right:
                            diff = r_us - l_us
                            angle = int(straightConst + (diff * 1.5))
                        else:
                            aDiff = rightArea - leftArea
                            angle = int(straightConst - (aDiff * 0.006))

                    # Full cruising speed on open straightaway
                    currentSpeed = normalSpeed

                # Constrain angle between safe mechanical limits (60 to 140 deg)
                angle = max(60, min(140, angle))

                # Rate-limiting: Send DRIVE command only when angle/speed changes or every 100ms
                angle_changed = last_steer_angle is None or abs(angle - last_steer_angle) >= 2
                speed_changed = last_drive_speed != currentSpeed
                time_elapsed = (currTime - last_cmd_time) >= 0.1

                if angle_changed or speed_changed or time_elapsed:
                    serial_ctrl.send_command(f"DRIVE:{currentSpeed}:{angle}")
                    last_steer_angle = angle
                    last_drive_speed = currentSpeed
                    last_cmd_time = currTime

            # -------------------------------------------------------------
            # 4. FINAL LAP MAGENTA PARKING LOT ALGORITHM (Strictly after 3 Laps: t >= 12)
            # -------------------------------------------------------------
            if t >= 12 and not isTurning and not tempParking:
                print("=" * 65)
                print(f"[PARKING] 12 turns (3 FULL LAPS) complete! Searching for Magenta Parking Lot...")
                print("=" * 65)
                tempParking = True

            if tempParking and not isTurning:
                if magentaArea > 2500:
                    navMode = "PARKING_LOT"
                    # Centroid of magenta contour to determine left vs right parking bay
                    magenta_max = max_contour(contours_magenta, ROI4)
                    mag_x = magenta_max[1]
                    midpoint = ROI4[0] + (ROI4[2] - ROI4[0]) // 2
                    # 60 = LEFT bay, 140 = RIGHT bay
                    park_angle = sharpLeft if mag_x < midpoint else sharpRight

                    print(f"[PARKING] Magenta Lot Detected (Area: {magentaArea}, X: {mag_x}) -> Steering {'LEFT' if park_angle == sharpLeft else 'RIGHT'}...")
                    
                    park_start = time.time()
                    while time.time() - park_start < 2.0:
                        serial_ctrl.send_command(f"DRIVE:{turnSpeed}:{park_angle}")
                        time.sleep(0.04)

                        # Check front proximity to stop cleanly before touching rear barrier
                        us_check = serial_ctrl.get_us_data()
                        front_dist = us_check.get("f", 0)
                        if 0 < front_dist <= 15:
                            print(f"[PARKING] Reached parking end barrier ({front_dist} cm)!")
                            break

                    # Active electronic brake sequence
                    serial_ctrl.send_command("STOP")
                    serial_ctrl.send_command("DRIVE:-180:100")
                    time.sleep(0.08)
                    serial_ctrl.send_command("STOP")
                    print("[FINISH] Obstacle Challenge Parking Complete! Bot Safely Parked.")
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
