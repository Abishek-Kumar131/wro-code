#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Obstacle Challenge (Round 2) - camera only.

Strategy follows the Canada team (canada-team/src/ObstacleChallengeV2.py), adapted to this car:
  No pillar  : PD steering on (right wall area - left wall area). Magenta counts as wall,
               so the car keeps away from the parking lot during the laps.
  Pillar     : the nearest pillar is steered to a target x (red -> RED_TARGET on the left
               of the image = pass it on the right; green -> GREEN_TARGET = pass on the left).
               PD on the x error plus a push that grows as the pillar gets closer (CY).
               Pillars that slide under the nose (END_CONST) are dropped so the car
               straightens instead of clipping them with a rear wheel.
  Too close  : a huge pillar right ahead -> short reverse, then continue.
  Turns      : the turn-colour floor line starts a turn; the turn ends (and is counted)
               when the turning side's wall fills its ROI again, or a pillar takes over
               once the line is out of view. A small corner ROI sharpens tight corners.
  Parking    : after 12 turns, pillars are all passed on the outside to stay near the outer
               wall, the magenta lot is found in the left or right ROI, and Canada's parking
               sequence drives into it and stops when the wall ahead fills the floor ROI.

Removed from Canada's code: the 2023-24 three-point turn (not a 2026 rule), the 5 s stop
before parking, Hiwonder LEDs/buzzer. Blocking sequences re-send commands so the ESP32
500 ms failsafe never interrupts them.

Usage
  python3 obstacle_challenge_R2.py                  wait for button, 3 laps + parking
  python3 obstacle_challenge_R2.py --no-display     competition mode
  python3 obstacle_challenge_R2.py --parking-left   test: start the parking lap now (lot on the left)
  python3 obstacle_challenge_R2.py --parking-right  test: start the parking lap now (lot on the right)
  python3 obstacle_challenge_R2.py --turns 4        test: park after 1 lap
  python3 obstacle_challenge_R2.py --steer-only     test: motor off, watch the steering react
  Other: --dir left|right, --webcam, --no-wait, --pin N, --active-high
"""

import argparse
import math
import sys
import time
import traceback

import cv2

from masks import PILLAR_GREEN_MIN_ASPECT, PILLAR_RED_MIN_ASPECT
from wro_serial import WROSerialController, Drive
from wro_functions import (CameraManager, FpsCounter, roi_hsv_lab, wall_mask, orange_mask, blue_mask,
                           red_mask, green_mask, magenta_mask, contours_of, max_contour,
                           draw_roi, draw_offset_contours, wait_for_button_press,
                           roi_px, area_norm, is_wide, apply_roi_config)

# ============================================================================ tuning
# Camera regions, stored as FRACTIONS of the frame (x1, y1, x2, y2, each 0..1) so they
# follow the capture resolution instead of being pinned to one frame size.
#
# There are two sets because a 16:9 frame shows more to the left and right than a 4:3
# crop of the same camera. A 4:3 crop covers the middle 75% of the 16:9 width, so an x
# fraction converts as  x_wide = 0.125 + 0.75 * x_43 . The y fractions are identical:
# cropping to 4:3 removes width, not height.
#
# Contour areas are normalised to "640x480 equivalent pixels", so every area threshold
# below keeps its meaning whatever resolution the camera gives.
# Pillar targets convert the same way. In 16:9 the wall ROIs take a full half of the frame
# each and the pillar ROI the full width, spending the recovered view where it helps most.
TARGETS_43 = {"red": 0.172, "green": 0.828, "red_min_x": 0.344, "green_max_x": 0.656}
TARGETS_WIDE = {"red": 0.254, "green": 0.746, "red_min_x": 0.383, "green_max_x": 0.617}

ROIS_43 = {
    "left":   (0.000, 0.365, 0.516, 0.552),
    "right":  (0.516, 0.365, 1.000, 0.552),
    "pillar": (0.094, 0.250, 0.906, 0.719),
    "floor":  (0.313, 0.542, 0.688, 0.646),
    "corner": (0.422, 0.250, 0.578, 0.292),
    "bumper": (0.300, 0.700, 0.700, 0.990),
}
# Wide defaults sit further out and are taller than the 4:3 ones: a wide lens puts the
# side walls near the edges of the frame. Starting point only - set them on the real
# track with:  python3 tune_rois.py --round 2
ROIS_WIDE = {
    "left":   (0.00, 0.36, 0.34, 0.58),
    "right":  (0.66, 0.36, 1.00, 0.58),
    "pillar": (0.00, 0.20, 1.00, 0.72),
    "floor":  (0.33, 0.62, 0.67, 0.80),
    "corner": (0.44, 0.24, 0.56, 0.30),
    "bumper": (0.28, 0.72, 0.72, 0.99),
}

# Steering. On this car 60 = full left, 100 = straight, 140 = full right.
SERVO_CENTER = 100
# Usable steering range: 100 = straight, 70 = full left, 130 = full right (+-30).
# Past about 30 deg the front tyres scrub and skid instead of steering.
SHARP_LEFT, SHARP_RIGHT = 70, 130
KP, KD = 0.015, 0.01                        # wall PD (no pillar)
PILLAR_GAINS = {"normal": (0.25, 0.25, 0.08, 40),     # (kp, kd, cy, end_const)
                "crowded": (0.20, 0.20, 0.05, 70)}    # 2+ pillars of one colour (inside corner)
MAX_DIST = 370              # pillars farther than this (px from bottom centre) are ignored

# Pillar size filters (contour area, px)
RED_MIN_AREA = 150
GREEN_MIN_AREA = 200
PILLAR_PREFILTER_AREA = 100
RED_TOO_CLOSE, GREEN_TOO_CLOSE = 6500, 8000     # reverse if a pillar this big is right ahead
# At contact range a sign is BELOW the pillar ROI, so its area there stops growing and can
# even fall back to zero - the car went blind exactly when it was about to hit something.
# The bumper box sits under the pillar box and watches what the car is driving into. Any
# sign-coloured mass this big in it counts as a block ahead, whatever the pillar box says.
BUMPER_BLOCK_AREA = 2500
PILLAR_SLOW_AREA = 3500     # ease off the throttle once a sign is this big: a slower bump
                            # draws far less stall current than a full-speed one
RED_SKIP_WALL, GREEN_SKIP_WALL = 11500, 12000   # ignore pillars while a wall fills its ROI this much

# Turns
LINE_MIN_AREA = 100
EXIT_THRESH = 4000          # wall area on the turning side that ends a turn
CORNER_AREA = {"left": 1000, "right": 1250}     # corner ROI area that forces a sharp turn
TURN_LINE_COOLDOWN = 1.0    # s after a counted turn during which the turn line is ignored
# A corner is only counted if it was really driven. Two guards, both learned the hard way:
# a jammed car sat on one floor line and clocked up 11 of its 12 turns in 80 seconds
# without moving, then stopped mid-round believing the run was over.
TURN_MIN_TIME = 0.6         # s a turn must last before it can be counted (a corner is slower)
LINE_REARM_TIME = 0.4       # s the line must be OUT of the floor ROI before it can start
                            # another turn - so the line is a crossing, not a level reading
TOTAL_TURNS = 12

# Parking lot: only avoided, never entered (touching its limitations ends the round, rule 9.24.7)
LOT_AVOID_AREA = 5000       # magenta this big in a side ROI -> steer away from the lot

# Finish: after the last corner, stop in the start section (same as the open challenge)
FINISH_DELAY = {"left": 1.0, "right": 1.5, "none": 1.25}
FINISH_STRAIGHT_TOL = 10    # steering must be within this of centre to start the finish timer
FINISH_MAX_WAIT = 3.0       # if it never settles that straight, stop anyway this long after the last turn
BRAKE_SPEED = -180

# Speed (PWM 0-255). This motor stalls below ~220.
SPEED = 225
REVERSE_SPEED = -230
DODGE_SPEED = 215           # slower while going round a block that was right in front
# Backing away from a block too close to steer round.
# This is closed loop: it reverses until the block has really shrunk out of the way, not
# for a fixed time. A fixed short reverse followed by a cooldown of full-speed forward was
# why the car appeared to freeze nose-to-nose with a block: it undid its own escape, crept
# forward again, and stalled against it.
REVERSE_MAX_TIME = 1.0      # s: hard cap on one attempt (stops it reversing into a wall)
REVERSE_ESCALATE = 1.6      # multiply that cap on a repeat attempt
REVERSE_CLEAR_FRAC = 0.55   # "clear" = the block shrank to this fraction of the too-close area
REVERSE_STEER = 15          # deg of counter-steer on a repeat, to come out aimed past it
# The escape always reverses from a standstill, usually pressed against the thing it is
# escaping - the hardest possible start. Reverse out of forward motion breaks away easily
# (which is why the R1 manoeuvre works), but from rest the gearbox stiction can swallow
# 230/255 completely and the car just sits there. So start every escape at full duty for a
# moment, then settle to REVERSE_SPEED. Check the real figure with:
#     python3 test_reverse.py --from-motion        (does it reverse out of motion?)
#     python3 test_reverse.py --kick 150           (does a full-duty kick start it from rest?)
REVERSE_KICK_SPEED = -255   # full duty, just to break away
REVERSE_KICK_TIME = 0.15    # s of it before settling to REVERSE_SPEED
REVERSE_STREAK_WINDOW = 3.0  # s: another reverse within this counts as the same jam
REVERSE_COOLDOWN = 0.5      # s before another reverse may start
DODGE_TIME = 1.3            # s of holding the steering that goes round the block
START_DELAY = 0.5

WINDOW = "WRO R2 Obstacle Challenge"


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
    p = argparse.ArgumentParser(description="WRO 2026 Obstacle Challenge (camera only)")
    p.add_argument("--no-display", action="store_true", help="no monitor window (competition)")
    p.add_argument("--no-wait", "--nowait", "--instant-start", action="store_true", help="start immediately")
    p.add_argument("--webcam", "-w", action="store_true", help="use USB webcam instead of Pi camera")
    p.add_argument("--pin", type=int, default=17, help="start button GPIO (BCM)")
    p.add_argument("--active-high", action="store_true", help="button wired to 3.3 V instead of GND")
    p.add_argument("--dir", choices=["left", "right"], help="force track direction")
    p.add_argument("--turns", type=int, default=TOTAL_TURNS, help="start parking after this many turns")
    p.add_argument("--steer-only", action="store_true", help="motor off; steering still reacts")
    p.add_argument("--narrow", action="store_true", help="capture 4:3 instead of full-width 16:9")
    p.add_argument("--swap-rb", dest="swap_rb", action="store_true", default=None,
                   help="force a red/blue swap (use if the picture looks blue)")
    p.add_argument("--no-swap-rb", dest="swap_rb", action="store_false",
                   help="never swap red/blue, even if the pixel format suggests it")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[CONFIG] Ignoring old/unknown arguments: {unknown}")
    return args


def find_pillar(contours, target, colour, best, roi, ctx):
    """
    Canada's pillar selection. Updates `best` with the nearest usable pillar of this colour.
    Returns (number of pillars in the counting band, area of a dangerously close pillar or 0).
    """
    count = 0
    too_close = 0
    sx, sy = 640.0 / ctx["fw"], 480.0 / ctx["fh"]     # frame px -> 640x480 equivalent
    for cnt in contours:
        area = cv2.contourArea(cnt) * ctx["area"]
        if colour == "red":
            if area <= RED_MIN_AREA:
                continue
        else:
            if area <= GREEN_MIN_AREA:
                continue

        x, y, w, h = cv2.boundingRect(cnt)
        x += roi[0] + w // 2
        y += roi[1] + h
        dist = round(math.dist([x * sx, y * sy], [320, 480]))

        if 160 < dist < 380:
            count += 1

        if True:
            limit, in_path = ((RED_TOO_CLOSE, x >= ctx["red_min_x"]) if colour == "red"
                              else (GREEN_TOO_CLOSE, x <= ctx["green_max_x"]))
            if area > limit and in_path:
                too_close = max(too_close, int(area))

        # Drop pillars that are passing under the nose, or while a wall fills the view
        if y > roi[3] - ctx["end_const"] / sy or dist > MAX_DIST:
            continue
        skip_wall = RED_SKIP_WALL if colour == "red" else GREEN_SKIP_WALL
        if ctx["left_area"] > skip_wall or ctx["right_area"] > skip_wall:
            continue

        if dist < best.dist:
            best.area, best.dist, best.x, best.y = area, dist, x, y
            best.target, best.w, best.h = target, w, h
    return count, too_close


def bumper_block(img, roi, geom, frac=1.0):
    """
    Anything sign-coloured filling the box right in front of the wheels.

    Everything in this box is by definition in the car's path, so there is no x test and
    no distance test - unlike close_pillar, which is looking further out.
    Returns (area, side): red is passed on its right, green on its left.
    """
    hsv, _ = roi_hsv_lab(img, roi)
    limit = BUMPER_BLOCK_AREA * frac
    worst, side = 0, None
    for colour, mask in (("red", red_mask(hsv)), ("green", green_mask(hsv))):
        area = int(max_contour(contours_of(mask, PILLAR_PREFILTER_AREA), roi)[0] * geom["area"])
        if area > limit and area > worst:
            worst, side = area, "right" if colour == "red" else "left"
    return worst, side


def close_pillar(img, roi, geom, frac=1.0):
    """
    Is a traffic sign right in front of the bumper?

    Returns (area, side) for the worst offender, or (0, None). `side` is the way the car
    has to go to get round it: red is passed on its right, green on its left.

    Deliberately separate from find_pillar: that one throws away pillars sitting low in
    the ROI or hidden behind a big wall reading, which is exactly what a block about to
    be hit looks like - so it reports P=0 for the very thing in the way. `frac` lowers the
    bar, so backing away can stop on a smaller area than the one that triggered it.
    """
    hsv, _ = roi_hsv_lab(img, roi)
    worst, side = 0, None
    for colour, contours, limit in (
            ("red", contours_of(red_mask(hsv), PILLAR_PREFILTER_AREA, PILLAR_RED_MIN_ASPECT),
             RED_TOO_CLOSE * frac),
            ("green", contours_of(green_mask(hsv), PILLAR_PREFILTER_AREA, PILLAR_GREEN_MIN_ASPECT),
             GREEN_TOO_CLOSE * frac)):
        for cnt in contours:
            area = cv2.contourArea(cnt) * geom["area"]
            if area <= limit or area <= worst:
                continue
            x, _, w, _ = cv2.boundingRect(cnt)
            x += roi[0] + w // 2
            in_path = (x >= geom["red_min_x"]) if colour == "red" else (x <= geom["green_max_x"])
            if in_path:
                worst, side = int(area), "right" if colour == "red" else "left"
    return worst, side


def main():
    args = parse_args()
    show = not args.no_display
    status_to_terminal = sys.stdout.isatty()

    print("=" * 65)
    print("   ROBOVANGUARD - WRO 2026 Obstacle Challenge (camera only, Canada strategy)")
    print("=" * 65)

    link = WROSerialController(auto_connect=False)
    link.connect(wait=5.0)
    link.send_command("STOP")
    drive = Drive(link)

    if show:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 800, 600)
        except Exception as e:
            print(f"[DISPLAY] No window available ({e}); continuing without display.")
            show = False

    camera = CameraManager(force_webcam=args.webcam, wide=not args.narrow, swap_rb=args.swap_rb)
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
    rois = apply_roi_config(ROIS_WIDE if wide else ROIS_43, wide)
    targets = TARGETS_WIDE if wide else TARGETS_43
    ROI_LEFT = roi_px(rois["left"], FRAME_W, FRAME_H)
    ROI_RIGHT = roi_px(rois["right"], FRAME_W, FRAME_H)
    ROI_CORNER = roi_px(rois["corner"], FRAME_W, FRAME_H)
    RED_TARGET = int(targets["red"] * FRAME_W)
    GREEN_TARGET = int(targets["green"] * FRAME_W)
    AREA = area_norm(FRAME_W, FRAME_H)
    GEOM = {"fw": FRAME_W, "fh": FRAME_H, "area": AREA,
            "red_min_x": targets["red_min_x"] * FRAME_W,
            "green_max_x": targets["green_max_x"] * FRAME_W}
    print(f"[CAMERA] {FRAME_W}x{FRAME_H} "
          f"({'16:9 - full sensor width' if wide else '4:3 - sides cropped off the sensor'})")
    print(f"[ROI] left {ROI_LEFT}  right {ROI_RIGHT}  corner {ROI_CORNER}")
    print(f"[ROI] pillar targets: red x={RED_TARGET}, green x={GREEN_TARGET}")

    # ------------------------------------------------------------------ run state
    red_target, green_target = RED_TARGET, GREEN_TARGET
    roi_pillar = roi_px(rois["pillar"], FRAME_W, FRAME_H)
    roi_floor = roi_px(rois["floor"], FRAME_W, FRAME_H)
    roi_bumper = roi_px(rois["bumper"], FRAME_W, FRAME_H)
    corner_on = False

    turn_dir = args.dir or "none"
    t = 0
    l_turn = r_turn = False
    line_cooldown_until = 0.0
    line_last_seen = 0.0                   # so a line already in view cannot re-trigger
    turn_started = 0.0
    prev_diff = 0
    prev_error = 0
    error = 0
    angle = SERVO_CENTER

    finish_at = None
    last_turn_time = 0.0
    reverse_ready_at = 0.0
    last_reverse_end = 0.0
    reverse_streak = 0                     # repeats of the same jam, to escalate the escape
    dodge_until = 0.0                      # hold the steering that goes round the block
    dodge_dir = None
    link_was_ok = True
    exit_reason = "unknown"

    if not args.no_wait:
        wait_for_button_press(args.pin, args.active_high, camera, show, WINDOW)

    warn = link.health_warning()
    if warn:
        print("=" * 66)
        print(f"[WARNING] Before starting: {warn}")
        print("=" * 66)
    time.sleep(START_DELAY)

    fps = FpsCounter()
    t_start = time.time()
    last_status = 0.0
    speed = 0 if args.steer_only else SPEED
    print(f"[GO] Driving. Stopping after {args.turns} turns. Direction: {turn_dir.upper()}")

    try:
        while True:
            img = camera.capture_array()
            if img is None:
                time.sleep(0.005)
                continue
            now = time.time()
            fps.tick()

            # ---------------------------------------------------------- vision
            hsv_l, lab_l = roi_hsv_lab(img, ROI_LEFT)
            hsv_r, lab_r = roi_hsv_lab(img, ROI_RIGHT)
            mag_l = magenta_mask(lab_l)
            mag_r = magenta_mask(lab_r)
            wall_l = wall_mask(hsv_l, lab_l)
            wall_r = wall_mask(hsv_r, lab_r)
            # magenta counts as wall so the car steers around the parking lot, never into it
            wall_l = cv2.bitwise_or(wall_l, mag_l)
            wall_r = cv2.bitwise_or(wall_r, mag_r)
            c_left = contours_of(wall_l, 100)
            c_right = contours_of(wall_r, 100)
            # areas normalised to 640x480-equivalent pixels so the thresholds above hold
            left_area = int(max_contour(c_left, ROI_LEFT)[0] * AREA)
            right_area = int(max_contour(c_right, ROI_RIGHT)[0] * AREA)
            lot_left_area = int(max_contour(contours_of(mag_l, 100), ROI_LEFT)[0] * AREA)
            lot_right_area = int(max_contour(contours_of(mag_r, 100), ROI_RIGHT)[0] * AREA)

            hsv_f, lab_f = roi_hsv_lab(img, roi_floor)
            orange_area = int(max_contour(contours_of(orange_mask(hsv_f, lab_f), LINE_MIN_AREA), roi_floor)[0] * AREA)
            blue_area = int(max_contour(contours_of(blue_mask(hsv_f, lab_f), LINE_MIN_AREA), roi_floor)[0] * AREA)

            hsv_p, _ = roi_hsv_lab(img, roi_pillar)
            c_red = contours_of(red_mask(hsv_p), PILLAR_PREFILTER_AREA, PILLAR_RED_MIN_ASPECT)
            c_green = contours_of(green_mask(hsv_p), PILLAR_PREFILTER_AREA, PILLAR_GREEN_MIN_ASPECT)

            corner_area = 0
            if corner_on:
                hsv_c, lab_c = roi_hsv_lab(img, ROI_CORNER)
                corner_area = int((max_contour(contours_of(wall_mask(hsv_c, lab_c), 50), ROI_CORNER)[0]
                                   + max_contour(contours_of(magenta_mask(lab_c), 50), ROI_CORNER)[0]) * AREA)

            # ---------------------------------------------------------- pillars
            ctx = dict(GEOM, left_area=left_area, right_area=right_area)
            scan = Pillar()
            # choose gains from how many pillars of one colour are in view
            n_g, _ = find_pillar(c_green, green_target, "green", scan, roi_pillar, dict(ctx, end_const=40))
            n_r, _ = find_pillar(c_red, red_target, "red", scan, roi_pillar, dict(ctx, end_const=40))
            gains = "crowded" if (n_g >= 2 or n_r >= 2) else "normal"
            c_kp, c_kd, c_y, end_const = PILLAR_GAINS[gains]
            ctx["end_const"] = end_const

            pillar = Pillar()
            find_pillar(c_green, green_target, "green", pillar, roi_pillar, ctx)
            find_pillar(c_red, red_target, "red", pillar, roi_pillar, ctx)

            # ---------------------------------------------------------- block right ahead
            close_area, close_side = close_pillar(img, roi_pillar, GEOM)
            bump_area, bump_side = bumper_block(img, roi_bumper, GEOM)
            if bump_area and not close_area:
                # too close for the pillar box to see it at all
                close_area, close_side = bump_area, bump_side
            if close_area and now >= reverse_ready_at:
                if now - last_reverse_end > REVERSE_STREAK_WINDOW:
                    reverse_streak = 0
                reverse_streak += 1
                if close_side:
                    dodge_dir = close_side
                # Reversing with the wheels turned one way swings the nose the other way,
                # so counter-steer to come out of it aimed past the block. Straight back on
                # the first attempt: that cannot put the car anywhere new.
                back_angle = SERVO_CENTER
                if reverse_streak >= 2 and dodge_dir in ("left", "right"):
                    lean = -REVERSE_STEER if dodge_dir == "right" else REVERSE_STEER
                    # every other attempt swings the wheels the other way: a wiggle breaks
                    # a jam that simply repeating the same move cannot
                    back_angle += lean if reverse_streak % 2 == 0 else -lean
                limit = REVERSE_MAX_TIME * (REVERSE_ESCALATE if reverse_streak >= 2 else 1.0)
                print(f"\n[BLOCK] {close_side or 'pillar'} {close_area}px right ahead "
                      f"(attempt {reverse_streak}) -> reverse at {back_angle} for up to "
                      f"{limit:.1f}s, then go {dodge_dir or 'by the walls'}")
                drive.drive(0, SERVO_CENTER, force=True)
                time.sleep(0.12)                 # let the gearbox stop before it reverses
                t_rev = time.time()
                cleared = False
                while time.time() - t_rev < limit:
                    # force=True resends every frame, so a dropped line or a reconnect
                    # cannot leave the car standing still on the ESP32 failsafe mid-escape
                    kicking = time.time() - t_rev < REVERSE_KICK_TIME
                    drive.drive(REVERSE_KICK_SPEED if kicking else REVERSE_SPEED,
                                back_angle, force=True)
                    f_rev = camera.capture_array()
                    if f_rev is None:
                        continue
                    if (close_pillar(f_rev, roi_pillar, GEOM, REVERSE_CLEAR_FRAC)[0] == 0
                            and bumper_block(f_rev, roi_bumper, GEOM, REVERSE_CLEAR_FRAC)[0] == 0):
                        cleared = True
                        break
                drive.drive(0, SERVO_CENTER, force=True)
                time.sleep(0.1)
                last_reverse_end = time.time()
                line_last_seen = last_reverse_end   # blind during the escape: do not re-arm
                reverse_ready_at = last_reverse_end + REVERSE_COOLDOWN
                dodge_until = last_reverse_end + DODGE_TIME
                print(f"[BLOCK] {'clear' if cleared else 'still there'} after "
                      f"{last_reverse_end - t_rev:.2f}s of reverse")
                continue

            # ---------------------------------------------------------- floor lines / turns
            if turn_dir == "none":
                if orange_area > LINE_MIN_AREA:
                    turn_dir = "right"
                elif blue_area > LINE_MIN_AREA:
                    turn_dir = "left"
                if turn_dir != "none":
                    print(f"[DIR] First line -> direction {turn_dir.upper()}")

            # `line_now` is the line being in the box. A turn starts on the car CROSSING it,
            # and only after the line has been out of the box for a while: standing still
            # over a line is one crossing, not a hundred.
            line_now = ((turn_dir == "right" and orange_area > LINE_MIN_AREA) or
                        (turn_dir == "left" and blue_area > LINE_MIN_AREA))
            if (line_now and now >= line_cooldown_until and now >= dodge_until
                    and now - line_last_seen >= LINE_REARM_TIME):
                turn_started = now
                if turn_dir == "right":
                    r_turn = True
                else:
                    l_turn = True
                if pillar.area != 0 and (
                        (left_area > 500 and turn_dir == "left") or (right_area > 500 and turn_dir == "right")):
                    corner_on = True
            if line_now:
                line_last_seen = now

            def end_turn(method):
                nonlocal l_turn, r_turn, t, prev_error, prev_diff, line_cooldown_until, last_turn_time
                if now - turn_started < TURN_MIN_TIME:
                    return          # too quick to be a corner: stay in the turn
                l_turn = r_turn = False
                prev_error = prev_diff = 0
                t += 1
                last_turn_time = now
                line_cooldown_until = now + TURN_LINE_COOLDOWN
                print(f"[TURN] {t}/{args.turns} done ({method}) at {now - t_start:.1f}s")

            # ---------------------------------------------------------- steering
            if True:
                if pillar.area == 0:
                    a_diff = right_area - left_area
                    angle = SERVO_CENTER - (a_diff * KP + (a_diff - prev_diff) * KD)
                    prev_diff = a_diff
                else:
                    if (l_turn or r_turn) and not line_now:
                        end_turn("pillar")
                    error = pillar.target - pillar.x
                    angle = SERVO_CENTER - (error * c_kp + (error - prev_error) * c_kd)
                    push = int(c_y * (pillar.y - roi_pillar[1]))
                    angle += push if error <= 0 else -push

                # corner ROI: sharp turn at tight corners
                if corner_area > CORNER_AREA.get(turn_dir, 1e9):
                    l_turn = r_turn = False
                    if pillar.area > 5000 or (turn_dir == "right" and pillar.area > 3500):
                        angle = SERVO_CENTER
                    else:
                        angle = SHARP_RIGHT if turn_dir == "right" else SHARP_LEFT
                if ((pillar.area == 0 and corner_area < 100)
                        or (abs(left_area - right_area) > 5000 and corner_area > 1000)):
                    corner_on = False

                # keep away from the parking lot
                if lot_right_area > LOT_AVOID_AREA:
                    angle = SHARP_LEFT
                elif lot_left_area > LOT_AVOID_AREA:
                    angle = SHARP_RIGHT

                # turn exit by wall, and default turn angles when no pillar is in view
                if ((r_turn and right_area >= EXIT_THRESH)
                        or (l_turn and left_area >= EXIT_THRESH)) and not line_now:
                    end_turn("wall")
                if r_turn and pillar.area == 0 and right_area < 5000:
                    angle = SHARP_RIGHT
                elif l_turn and pillar.area == 0 and left_area < 5000:
                    angle = SHARP_LEFT

            # Just backed away from a block: hold the steering that goes round it. Without
            # this the block is usually too close to be tracked (P reads 0) and the wall PD
            # drives straight back into it - the other half of the freeze.
            if now < dodge_until and dodge_dir in ("left", "right") and (
                    lot_left_area <= LOT_AVOID_AREA and lot_right_area <= LOT_AVOID_AREA):
                angle = SHARP_LEFT if dodge_dir == "left" else SHARP_RIGHT

            angle = int(max(SHARP_LEFT, min(SHARP_RIGHT, angle)))

            # ------------------------------------------------ finish (no parking)
            # After the last corner: once the steering is straight, drive on briefly and stop
            # in the start section, exactly like the open challenge.
            if t >= args.turns and finish_at is None:
                if not (l_turn or r_turn) and abs(angle - SERVO_CENTER) <= FINISH_STRAIGHT_TOL:
                    finish_at = now + FINISH_DELAY[turn_dir]
                    print(f"[FINISH] All {t} turns done. Stopping in {FINISH_DELAY[turn_dir]:.2f}s")
                elif now - last_turn_time > FINISH_MAX_WAIT:
                    finish_at = now      # steering never settled straight: stop anyway
                    print("[FINISH] Steering never settled straight - stopping now")
            if finish_at is not None and now >= finish_at:
                drive.stop(0 if args.steer_only else BRAKE_SPEED, SERVO_CENTER)
                exit_reason = f"finished {t} turns"
                break

            prev_error = error
            # Ease off for anything close: a slow nudge is a bump, a fast one is a stall,
            # and a stalled motor is what browns the ESP32 out.
            speed_now = (DODGE_SPEED if (now < dodge_until or bump_area
                                         or pillar.area > PILLAR_SLOW_AREA) else speed)
            drive.drive(0 if args.steer_only else speed_now, angle)

            if link.link_ok != link_was_ok:
                link_was_ok = link.link_ok
                print(f"[LINK] {'restored' if link_was_ok else 'DOWN - car stopped by ESP32 failsafe'}")

            # ---------------------------------------------------------- debug output
            if finish_at is not None:
                state = "FINISHING"
            elif now < dodge_until and dodge_dir:
                state = f"DODGE-{dodge_dir[0].upper()}"
            elif pillar.area:
                state = "PILLAR-R" if pillar.target == RED_TARGET else "PILLAR-G"
            else:
                state = "TURN" if (l_turn or r_turn) else "WALLS"

            if status_to_terminal and now - last_status > 0.2:
                last_status = now
                print(f"\r{state:10s} t={t:2d} L={left_area:5d} R={right_area:5d} P={int(pillar.area):5d} "
                      f"corner={corner_area:4d} ang={angle:3d} fps={fps.fps:4.1f} "
                      f"link={'ok' if link.link_ok else 'DOWN'}   ", end="", flush=True)

            if show:
                disp = img.copy()
                cv2.rectangle(disp, (roi_bumper[0], roi_bumper[1]), (roi_bumper[2], roi_bumper[3]),
                              (0, 0, 255) if bump_area else (120, 120, 120), 2)
                for roi, col in ((ROI_LEFT, (0, 255, 255)), (ROI_RIGHT, (0, 255, 255)),
                                 (roi_pillar, (255, 204, 0)), (roi_floor, (255, 0, 255))):
                    draw_roi(disp, roi, col)
                if corner_on:
                    draw_roi(disp, ROI_CORNER, (0, 0, 255))
                draw_offset_contours(disp, c_left, ROI_LEFT, (0, 255, 0))
                draw_offset_contours(disp, c_right, ROI_RIGHT, (0, 255, 0))
                draw_offset_contours(disp, c_red, roi_pillar, (0, 0, 255))
                draw_offset_contours(disp, c_green, roi_pillar, (0, 255, 0))
                if pillar.area:
                    cv2.circle(disp, (int(pillar.x), int(pillar.y)), 6, (255, 255, 255), -1)
                    cv2.line(disp, (pillar.target, 0), (pillar.target, 479), (255, 255, 255), 1)
                cv2.putText(disp, f"{state} dir:{turn_dir} turns:{t}/{args.turns} fps:{fps.fps:.0f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 204), 2)
                cv2.putText(disp, f"L:{left_area} R:{right_area} P:{int(pillar.area)} ang:{angle}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(disp, "link OK" if link.link_ok else "LINK DOWN", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if link.link_ok else (0, 0, 255), 2)
                cv2.imshow(WINDOW, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    exit_reason = "stopped from display window"
                    break

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
        print(f"[EXIT] Run ended: {exit_reason} | turns {t}/{args.turns} | "
              f"{time.time() - t_start:.1f}s | avg fps {fps.fps:.1f}")
        link.disconnect()


if __name__ == "__main__":
    main()
