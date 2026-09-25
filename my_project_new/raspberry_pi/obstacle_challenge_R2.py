#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Obstacle Challenge (Round 2) - camera + 6 ultrasonic sensors (hybrid).

This keeps the original ROBOVANGUARD strategy:
  Pillars  PD steering onto a target x for the nearest pillar (red -> pass on the right,
           green -> pass on the left), with a push that grows as the pillar gets closer,
           a short hold after it leaves view so the rear wheel clears it, and side
           ultrasonic guards so dodging never puts the car into a wall.
  Walls    When no pillar is in view: wall-area centering, or side ultrasonic centering
           when the walls are not visible.
  Turns    First floor line colour locks the direction. A corner starts when the inner
           wall's ROI empties and ends when the camera re-acquires a wall (0.8-2.2 s).
  Finish   After 12 corners, stop in the start section exactly like the open challenge
           (floor marker, the start baseline distances, the front wall, or a timeout).
           There is no parking: the lot is only avoided, never entered.
  Any time An obstacle within a few cm triggers a short reverse.

Fixed since the version on main:
  - endConst is now actually used. It was computed every frame and never applied, so a
    pillar kept steering the car after it had passed the nose - which is how a rear wheel
    clips a block and loses the 10-point "no signs moved" bonus.
  - A corner is only counted with the floor marker or a front-sensor confirmation.
  - 0 cm now means "nothing in range" instead of "collision", and proximity must repeat
    on two updates before it acts.
  - Reverse no longer blocks the loop blind; it keeps reading the camera and keeps the
    ESP32 failsafe fed.
  - Parking removed: the car now finishes like the open challenge. Magenta still counts
    as wall, so the car steers around the parking lot instead of into it.
  - Steering limited to 75-125 degrees to match the mechanical range of this linkage.
  - Any exception now stops the car and prints why, instead of only Ctrl+C.

Usage
  python3 obstacle_challenge_R2.py                 wait for button, 3 laps then stop
  python3 obstacle_challenge_R2.py --no-display    competition mode
  python3 obstacle_challenge_R2.py --turns 4       test: stop after 1 lap
  python3 obstacle_challenge_R2.py --steer-only    test: motor off, watch steering and sensors
  Other: --dir left|right, --webcam, --no-wait, --pin N, --active-high, --no-us
"""

import argparse
import math
import sys
import time
import traceback

import cv2

from masks import PILLAR_GREEN_MIN_ASPECT, PILLAR_RED_MIN_ASPECT
from wro_serial import WROSerialController, Drive
from wro_functions import (CameraManager, FpsCounter, Ultrasonics, roi_hsv_lab, wall_mask,
                           orange_mask, blue_mask, red_mask, green_mask, magenta_mask,
                           contours_of, max_contour, draw_roi, draw_offset_contours,
                           wait_for_button_press, roi_px, area_norm, is_wide)

# ============================================================================ tuning
# Pillar targets and camera regions, stored as FRACTIONS of the frame so they follow the
# capture resolution instead of being pinned to one frame size.
#
# Two sets, because a 16:9 frame shows more to the left and right than a 4:3 crop of the
# same camera. A 4:3 crop covers the middle 75% of the 16:9 width, so an x fraction
# converts as  x_wide = 0.125 + 0.75 * x_43 . The y fractions are identical: cropping to
# 4:3 removes width, not height.
#
# Contour areas and pillar distances are normalised to "640x480 equivalent", so every
# threshold below keeps its meaning whatever resolution the camera gives.

# where a pillar is steered to: red toward the left of the image (pass it on the right)
TARGETS_43 = {"red": 0.172, "green": 0.828}
TARGETS_WIDE = {"red": 0.254, "green": 0.746}

ROIS_43 = {
    "left":   (0.031, 0.354, 0.375, 0.458),
    "right":  (0.625, 0.354, 0.969, 0.458),
    "pillar": (0.000, 0.125, 1.000, 0.583),
    "floor":  (0.313, 0.563, 0.688, 0.708),
}
ROIS_WIDE = {
    "left":   (0.148, 0.354, 0.406, 0.458),
    "right":  (0.594, 0.354, 0.852, 0.458),
    "pillar": (0.000, 0.125, 1.000, 0.583),   # full width: spend the extra view on pillars
    "floor":  (0.359, 0.563, 0.641, 0.708),
}

# Steering. On this car 60 = full left, 100 = straight, 140 = full right.
SERVO_CENTER = 100
# Mechanical steering range of this car: 100 = straight, 75 = full left, 125 = full right
SERVO_MIN, SERVO_MAX = 75, 125

# Pillar avoidance
PILLAR_GAINS = {"normal": (0.34, 0.26, 0.14, 20),    # (kp, kd, cy, end_const)
                "crowded": (0.30, 0.22, 0.10, 40)}   # 2+ pillars of one colour
PILLAR_MIN_AREA = 70
PILLAR_MAX_DIST = 480
EVADE_HOLD = 0.70           # s to hold the evasion heading after the pillar leaves view
SIDE_GUARD_CM = 12          # pull back toward centre if a wall gets this close while dodging

# Wall centering
DUAL_WALL_GAIN = 0.012
WALL_VISIBLE_AREA = 120
CORNER_APPROACH_GAIN = 0.008
CORNER_APPROACH_TARGET = 600
CORNER_APPROACH_CLAMP = (92, 108)
US_CENTER_GAIN = 1.5
SINGLE_WALL_GAIN = 0.006

# Speed (PWM 0-255)
SPEED = 245
PILLAR_SPEED = 228
TURN_SPEED = 235
RETURN_SPEED = 230          # controlled speed while creeping to the finish
BRAKE_SPEED = -180
START_DELAY = 0.5

# Turns
TURN_THRESH = 150
WALL_REACQUIRE_AREA = 400
MIN_TURN_TIME = 0.8
MAX_TURN_TIME = 2.2
TURN_COOLDOWN = 3.0
POST_TURN_LINE_LOCKOUT = 0.6  # short, so the next corner marker is not swallowed
LINE_MIN_AREA = 120
FRONT_CORNER_CM = 45
TOTAL_TURNS = 12

# Ultrasonic safety
FRONT_HIT_CM = 12
SIDE_JAM_CM = 5
REAR_LIMIT_CM = 8
REVERSE_SPEED = -235
REVERSE_MIN_TIME = 0.35
REVERSE_MAX_TIME = 0.70
REVERSE_CLEAR_CM = 18
REVERSE_COOLDOWN = 1.2

# Finish: stopping in the start section after the last corner (same rules as the open challenge)
MIN_CLEAR_OF_CORNER = 0.5
FINISH_MARKER_AREA = 150
BASELINE_TOLERANCE_CM = 4.0
FRONT_WALL_STOP_CM = 25.0
HOME_TIMEOUT = 4.0

WINDOW = "WRO R2 Obstacle Challenge (hybrid)"


class Pillar:
    def __init__(self):
        self.area = 0
        self.dist = 1_000_000
        self.x = 0
        self.y = 0
        self.target = 0
        self.w = 0
        self.h = 0


def parse_args():
    p = argparse.ArgumentParser(description="WRO 2026 Obstacle Challenge (camera + ultrasonics)")
    p.add_argument("--no-display", action="store_true", help="no monitor window (competition)")
    p.add_argument("--no-wait", "--nowait", "--instant-start", action="store_true", help="start immediately")
    p.add_argument("--webcam", "-w", action="store_true", help="use USB webcam instead of Pi camera")
    p.add_argument("--pin", type=int, default=17, help="start button GPIO (BCM)")
    p.add_argument("--active-high", action="store_true", help="button wired to 3.3 V instead of GND")
    p.add_argument("--dir", choices=["left", "right"], help="force track direction")
    p.add_argument("--turns", type=int, default=TOTAL_TURNS, help="park after this many turns")
    p.add_argument("--steer-only", action="store_true", help="motor off; steering still reacts")
    p.add_argument("--no-us", action="store_true", help="ignore the ultrasonics")
    p.add_argument("--narrow", action="store_true", help="capture 4:3 instead of full-width 16:9")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[CONFIG] Ignoring old/unknown arguments: {unknown}")
    return args


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def find_pillar(contours, target, colour, best, end_const, roi, fw, fh, area_scale):
    """
    Picks the nearest usable pillar of one colour. Pillars that have slid below
    (ROI bottom - end_const) are dropped, so a block the car has already passed stops
    pulling the steering - that drop was missing on main.
    """
    count = 0
    sx, sy = 640.0 / fw, 480.0 / fh      # frame px -> 640x480 equivalent
    end_px = end_const / sy
    for cnt in contours:
        area = cv2.contourArea(cnt) * area_scale
        if area < PILLAR_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x += roi[0] + w // 2
        y += roi[1] + h
        dist = round(math.dist([x * sx, y * sy], [320, 480]))

        if 80 < dist < PILLAR_MAX_DIST:
            count += 1
        if dist > PILLAR_MAX_DIST:
            continue
        if y > roi[3] - end_px:     # passing under the nose: stop tracking it
            continue

        if dist < best.dist:
            best.area, best.dist, best.x, best.y = area, dist, x, y
            best.target, best.w, best.h = target, w, h
    return count


def main():
    args = parse_args()
    show = not args.no_display
    status_to_terminal = sys.stdout.isatty()
    use_us = not args.no_us

    print("=" * 65)
    print("   ROBOVANGUARD - WRO 2026 Obstacle Challenge (camera + 6 ultrasonics)")
    print("=" * 65)

    link = WROSerialController(auto_connect=False)
    link.connect(wait=5.0)
    link.send_command("STOP")
    drive = Drive(link)
    us = Ultrasonics(link, confirm=2)

    if show:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 800, 600)
        except Exception as e:
            print(f"[DISPLAY] No window available ({e}); continuing without display.")
            show = False

    camera = CameraManager(force_webcam=args.webcam, wide=not args.narrow)
    camera.start()
    probe = None
    for _ in range(15):            # let auto-exposure settle, and learn the frame size
        f = camera.capture_array()
        if f is not None:
            probe = f
    if probe is None:
        print("[ERROR] The camera returned no frames. Run test_camera_fov.py to check it.")
        link.disconnect()
        return

    FRAME_H, FRAME_W = probe.shape[:2]
    wide = is_wide(FRAME_W, FRAME_H)
    rois = ROIS_WIDE if wide else ROIS_43
    targets = TARGETS_WIDE if wide else TARGETS_43
    ROI_LEFT = roi_px(rois["left"], FRAME_W, FRAME_H)
    ROI_RIGHT = roi_px(rois["right"], FRAME_W, FRAME_H)
    ROI_PILLAR = roi_px(rois["pillar"], FRAME_W, FRAME_H)
    ROI_FLOOR = roi_px(rois["floor"], FRAME_W, FRAME_H)
    RED_TARGET = int(targets["red"] * FRAME_W)
    GREEN_TARGET = int(targets["green"] * FRAME_W)
    AREA = area_norm(FRAME_W, FRAME_H)
    print(f"[CAMERA] {FRAME_W}x{FRAME_H} "
          f"({'16:9 - full sensor width' if wide else '4:3 - sides cropped off the sensor'})")
    print(f"[ROI] left {ROI_LEFT}  right {ROI_RIGHT}  pillars {ROI_PILLAR}  floor {ROI_FLOOR}")
    print(f"[ROI] pillar targets: red x={RED_TARGET}, green x={GREEN_TARGET}")

    if not args.no_wait:
        wait_for_button_press(args.pin, args.active_high, camera, show, WINDOW)

    # ------------------------------------------------- baseline for the finish detection
    us.update()
    baseline = dict(us.values)
    print(f"[START] Baseline distances: {us.text()}")
    time.sleep(START_DELAY)

    # ---------------------------------------------------------------- run state
    turn_dir = args.dir or "none"
    turns = 0
    is_turning = False
    turn_start = 0.0
    marker_seen = False
    line_lockout_until = 0.0
    cooldown_until = 0.0
    reverse_ready_at = 0.0
    returning_home = False
    corner12_time = 0.0
    evade_until = 0.0
    evade_angle = SERVO_CENTER
    evade_target = None
    prev_error = 0
    angle = SERVO_CENTER
    link_was_ok = True
    exit_reason = "unknown"

    fps = FpsCounter()
    t_start = time.time()
    last_status = 0.0
    print(f"[GO] Driving. Parking after {args.turns} turns. Direction: {turn_dir.upper()}")

    try:
        while True:
            img = camera.capture_array()
            if img is None:
                time.sleep(0.005)
                continue
            now = time.time()
            fps.tick()
            if use_us:
                us.update()

            # ---------------------------------------------------------- vision
            hsv, lab = roi_hsv_lab(img, ROI_LEFT)
            # magenta counts as wall so the car steers around the parking lot, never into it
            c_left = contours_of(cv2.bitwise_or(wall_mask(hsv, lab), magenta_mask(lab)), 50)
            hsv, lab = roi_hsv_lab(img, ROI_RIGHT)
            c_right = contours_of(cv2.bitwise_or(wall_mask(hsv, lab), magenta_mask(lab)), 50)
            hsv_f, lab_f = roi_hsv_lab(img, ROI_FLOOR)
            c_orange = contours_of(orange_mask(hsv_f, lab_f), LINE_MIN_AREA)
            c_blue = contours_of(blue_mask(hsv_f, lab_f), LINE_MIN_AREA)
            hsv_p, _ = roi_hsv_lab(img, ROI_PILLAR)
            c_red = contours_of(red_mask(hsv_p), PILLAR_MIN_AREA - 1, PILLAR_RED_MIN_ASPECT)
            c_green = contours_of(green_mask(hsv_p), PILLAR_MIN_AREA - 1, PILLAR_GREEN_MIN_ASPECT)

            # areas normalised to 640x480-equivalent pixels so the thresholds above hold
            left_area = int(max_contour(c_left, ROI_LEFT)[0] * AREA)
            right_area = int(max_contour(c_right, ROI_RIGHT)[0] * AREA)
            orange_area = int(max_contour(c_orange, ROI_FLOOR)[0] * AREA)
            blue_area = int(max_contour(c_blue, ROI_FLOOR)[0] * AREA)

            # ---------------------------------------------------------- pillars
            scan = Pillar()
            n_g = find_pillar(c_green, GREEN_TARGET, "green", scan, 20, ROI_PILLAR, FRAME_W, FRAME_H, AREA)
            n_r = find_pillar(c_red, RED_TARGET, "red", scan, 20, ROI_PILLAR, FRAME_W, FRAME_H, AREA)
            gains = "crowded" if (n_g >= 2 or n_r >= 2) else "normal"
            c_kp, c_kd, c_y, end_const = PILLAR_GAINS[gains]

            pillar = Pillar()
            find_pillar(c_green, GREEN_TARGET, "green", pillar, end_const, ROI_PILLAR, FRAME_W, FRAME_H, AREA)
            find_pillar(c_red, RED_TARGET, "red", pillar, end_const, ROI_PILLAR, FRAME_W, FRAME_H, AREA)

            # ---------------------------------------------------------- emergency reverse
            if use_us and not is_turning and now >= reverse_ready_at:
                front_hit = (us.near("f", FRONT_HIT_CM) or us.near("f1", FRONT_HIT_CM)
                             or us.near("f2", FRONT_HIT_CM))
                side_jam = us.near("l", SIDE_JAM_CM) or us.near("r", SIDE_JAM_CM)
                if front_hit or (side_jam and pillar.area > 2000):
                    rev_steer = SERVO_CENTER
                    if us.near("l", 6) or left_area > 800:
                        rev_steer = SERVO_CENTER - 8     # obstacle on the left: nose away from it
                    elif us.near("r", 6) or right_area > 800:
                        rev_steer = SERVO_CENTER + 8
                    print(f"[REVERSE] Obstacle close ({us.text()}) -> backing off at {rev_steer} deg")

                    rev_start = time.time()
                    while time.time() - rev_start < REVERSE_MAX_TIME:
                        camera.capture_array()
                        us.update()
                        drive.drive(0 if args.steer_only else REVERSE_SPEED, rev_steer)
                        if us.near("b", REAR_LIMIT_CM):
                            break
                        if time.time() - rev_start >= REVERSE_MIN_TIME and us.clear_ahead(REVERSE_CLEAR_CM):
                            break
                        time.sleep(0.01)

                    reverse_ready_at = time.time() + REVERSE_COOLDOWN
                    cooldown_until = max(cooldown_until, reverse_ready_at)
                    continue

            # ---------------------------------------------------------- floor markers
            if not returning_home and now >= line_lockout_until:
                if turn_dir == "none":
                    if orange_area > LINE_MIN_AREA and orange_area >= blue_area:
                        turn_dir, marker_seen = "right", True
                        line_lockout_until = now + 1.2
                        print(f"[DIR] First line ORANGE ({orange_area} px) -> direction RIGHT (locked)")
                    elif blue_area > LINE_MIN_AREA:
                        turn_dir, marker_seen = "left", True
                        line_lockout_until = now + 1.2
                        print(f"[DIR] First line BLUE ({blue_area} px) -> direction LEFT (locked)")
                elif ((turn_dir == "right" and orange_area > LINE_MIN_AREA) or
                      (turn_dir == "left" and blue_area > LINE_MIN_AREA)):
                    marker_seen = True
                    line_lockout_until = now + 1.2

            # ---------------------------------------------------------- turns
            if is_turning:
                target = SERVO_MIN if turn_dir == "left" else SERVO_MAX
                elapsed = now - turn_start
                reacquired = elapsed >= MIN_TURN_TIME and (left_area >= WALL_REACQUIRE_AREA
                                                           or right_area >= WALL_REACQUIRE_AREA)
                if reacquired or elapsed >= MAX_TURN_TIME:
                    is_turning = False
                    turns += 1
                    cooldown_until = now + TURN_COOLDOWN
                    line_lockout_until = now + POST_TURN_LINE_LOCKOUT
                    marker_seen = False
                    why = "wall re-acquired" if reacquired else "max time"
                    print(f"[TURN] {turns}/{args.turns} {turn_dir.upper()} done ({why}, {elapsed:.2f}s)")
                    if turns >= args.turns and not returning_home:
                        returning_home = True
                        corner12_time = now
                        print(f"[FINISH] Last corner cleared - creeping to the finish at {RETURN_SPEED}")
                else:
                    angle = target
                    drive.drive(0 if args.steer_only else TURN_SPEED, target)

            elif not returning_home and turns < args.turns and now >= cooldown_until:
                wall_dropped = ((left_area <= TURN_THRESH and right_area <= TURN_THRESH)
                                or (turn_dir == "left" and left_area <= TURN_THRESH)
                                or (turn_dir == "right" and right_area <= TURN_THRESH))
                confirmed = marker_seen or (use_us and us.near("f", FRONT_CORNER_CM))
                if wall_dropped and confirmed and turn_dir != "none":
                    is_turning = True
                    turn_start = now
                    angle = SERVO_MIN if turn_dir == "left" else SERVO_MAX
                    why = "marker" if marker_seen else "front wall"
                    print(f"[TURN] Corner {turns + 1} starting ({why}; L={left_area} R={right_area})")
                    drive.drive(0 if args.steer_only else TURN_SPEED, angle)

            # ---------------------------------------------------------- finish
            # Same rules as the open challenge: stop in the start section.
            if returning_home and not is_turning:
                since_corner = now - corner12_time
                reasons = []
                marker = ((turn_dir == "right" and orange_area > FINISH_MARKER_AREA) or
                          (turn_dir == "left" and blue_area > FINISH_MARKER_AREA) or
                          (turn_dir == "none" and max(orange_area, blue_area) > FINISH_MARKER_AREA))
                if since_corner >= MIN_CLEAR_OF_CORNER and marker:
                    reasons.append("start/finish marker")
                if use_us and since_corner >= MIN_CLEAR_OF_CORNER:
                    for key in ("b", "f"):
                        base = baseline.get(key, 0)
                        if base > 0 and us.valid(key) and abs(us.get(key) - base) <= BASELINE_TOLERANCE_CM:
                            reasons.append(f"{key.upper()} back to baseline ({us.get(key)}/{base} cm)")
                            break
                if use_us and us.valid("f") and us.get("f") <= FRONT_WALL_STOP_CM:
                    reasons.append(f"front wall {us.get('f')} cm")
                if since_corner >= HOME_TIMEOUT:
                    reasons.append("timeout")

                if reasons:
                    print(f"[FINISH] Stopping: {', '.join(reasons)} ({since_corner:.2f}s after the last corner)")
                    drive.stop(0 if args.steer_only else BRAKE_SPEED, SERVO_CENTER)
                    exit_reason = f"finished {turns} turns"
                    break

            # ---------------------------------------------------------- steering
            if not is_turning:
                if pillar.area > 0 or (now < evade_until and evade_target is not None):
                    if pillar.area > 0:
                        evade_target = pillar.target
                        evade_until = now + EVADE_HOLD
                        error = pillar.target - pillar.x
                        angle = SERVO_CENTER - (error * c_kp) - ((error - prev_error) * c_kd)
                        push = int(c_y * (pillar.y - ROI_PILLAR[1]))
                        angle += push if error <= 0 else -push
                        if use_us:      # dodging must not put us into a wall
                            if pillar.target == RED_TARGET and us.near("r", SIDE_GUARD_CM):
                                angle = min(angle, 120)
                            elif pillar.target == GREEN_TARGET and us.near("l", SIDE_GUARD_CM):
                                angle = max(angle, 80)
                        prev_error = error
                        evade_angle = angle
                    else:
                        # pillar just left view: hold a parallel heading so the rear wheel clears it
                        if evade_target == RED_TARGET:
                            angle = clamp(evade_angle, 100, 115)
                        else:
                            angle = clamp(evade_angle, 85, 100)
                    speed = PILLAR_SPEED
                else:
                    both_walls = left_area > WALL_VISIBLE_AREA and right_area > WALL_VISIBLE_AREA
                    if both_walls:
                        angle = SERVO_CENTER - (right_area - left_area) * DUAL_WALL_GAIN
                    elif turn_dir == "right" and right_area <= WALL_VISIBLE_AREA:
                        angle = SERVO_CENTER - (left_area - CORNER_APPROACH_TARGET) * CORNER_APPROACH_GAIN
                        angle = clamp(angle, *CORNER_APPROACH_CLAMP)
                    elif turn_dir == "left" and left_area <= WALL_VISIBLE_AREA:
                        angle = SERVO_CENTER + (right_area - CORNER_APPROACH_TARGET) * CORNER_APPROACH_GAIN
                        angle = clamp(angle, *CORNER_APPROACH_CLAMP)
                    elif use_us and us.valid("l") and us.valid("r") and us.get("l") < 80 and us.get("r") < 80:
                        angle = SERVO_CENTER + (us.get("r") - us.get("l")) * US_CENTER_GAIN
                    else:
                        angle = SERVO_CENTER - (right_area - left_area) * SINGLE_WALL_GAIN
                    speed = RETURN_SPEED if returning_home else SPEED

                angle = int(clamp(angle, SERVO_MIN, SERVO_MAX))
                drive.drive(0 if args.steer_only else speed, angle)

            if link.link_ok != link_was_ok:
                link_was_ok = link.link_ok
                print(f"[LINK] {'restored' if link_was_ok else 'DOWN - car stopped by ESP32 failsafe'}")

            # ---------------------------------------------------------- debug output
            if is_turning:
                state = f"TURN-{turn_dir[0].upper()}"
            elif returning_home:
                state = "RETURN"
            elif pillar.area:
                state = "PILLAR-R" if pillar.target == RED_TARGET else "PILLAR-G"
            elif now < evade_until:
                state = "EVADING"
            else:
                state = "WALLS"

            if status_to_terminal and now - last_status > 0.2:
                last_status = now
                print(f"\r{state:10s} t={turns:2d} L={left_area:5d} R={right_area:5d} "
                      f"P={int(pillar.area):5d} ang={angle:3d} {us.text()} "
                      f"fps={fps.fps:4.1f} link={'ok' if link.link_ok else 'DOWN'}   ", end="", flush=True)

            if show:
                disp = img.copy()
                for roi, col in ((ROI_LEFT, (0, 255, 255)), (ROI_RIGHT, (0, 255, 255)),
                                 (ROI_PILLAR, (255, 204, 0)), (ROI_FLOOR, (255, 0, 255))):
                    draw_roi(disp, roi, col)
                draw_offset_contours(disp, c_left, ROI_LEFT, (0, 255, 0))
                draw_offset_contours(disp, c_right, ROI_RIGHT, (0, 255, 0))
                draw_offset_contours(disp, c_red, ROI_PILLAR, (0, 0, 255))
                draw_offset_contours(disp, c_green, ROI_PILLAR, (0, 255, 0))
                if pillar.area:
                    cv2.circle(disp, (int(pillar.x), int(pillar.y)), 6, (255, 255, 255), -1)
                    cv2.line(disp, (pillar.target, 0), (pillar.target, FRAME_H - 1), (255, 255, 255), 1)
                cv2.putText(disp, f"{state} dir:{turn_dir} turns:{turns}/{args.turns} fps:{fps.fps:.0f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 204), 2)
                cv2.putText(disp, f"L:{left_area} R:{right_area} P:{int(pillar.area)} ang:{angle}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(disp, us.text(), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(disp, "link OK" if link.link_ok else "LINK DOWN", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if link.link_ok else (0, 0, 255), 2)
                cv2.imshow(WINDOW, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    exit_reason = "stopped from display window"
                    break
                if key == ord('l'):
                    turn_dir = "left"
                elif key == ord('r'):
                    turn_dir = "right"

    except KeyboardInterrupt:
        exit_reason = "Ctrl+C"
    except Exception:
        exit_reason = "CRASH (traceback above)"
        traceback.print_exc()
    finally:
        link.send_command("STOP")
        camera.stop()
        if show:
            cv2.destroyAllWindows()
        print()
        print(f"[EXIT] Run ended: {exit_reason} | turns {turns}/{args.turns} | "
              f"{time.time() - t_start:.1f}s | avg fps {fps.fps:.1f}")
        link.disconnect()


if __name__ == "__main__":
    main()
