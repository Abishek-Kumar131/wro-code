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
                           roi_px, area_norm, is_wide)

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
}
ROIS_WIDE = {
    "left":   (0.000, 0.365, 0.500, 0.552),
    "right":  (0.500, 0.365, 1.000, 0.552),
    "pillar": (0.000, 0.250, 1.000, 0.719),
    "floor":  (0.359, 0.542, 0.641, 0.646),
    "corner": (0.441, 0.250, 0.559, 0.292),
}

# Steering. On this car 60 = full left, 100 = straight, 140 = full right.
SERVO_CENTER = 100
# Mechanical steering range of this car: 100 = straight, 75 = full left, 125 = full right
SHARP_LEFT, SHARP_RIGHT = 75, 125
KP, KD = 0.015, 0.01                        # wall PD (no pillar)
PILLAR_GAINS = {"normal": (0.25, 0.25, 0.08, 40),     # (kp, kd, cy, end_const)
                "crowded": (0.20, 0.20, 0.05, 70)}    # 2+ pillars of one colour (inside corner)
MAX_DIST = 370              # pillars farther than this (px from bottom centre) are ignored

# Pillar size filters (contour area, px)
RED_MIN_AREA = 150
GREEN_MIN_AREA = 200
PILLAR_PREFILTER_AREA = 100
RED_TOO_CLOSE, GREEN_TOO_CLOSE = 6500, 8000     # reverse if a pillar this big is right ahead
RED_SKIP_WALL, GREEN_SKIP_WALL = 11500, 12000   # ignore pillars while a wall fills its ROI this much

# Turns
LINE_MIN_AREA = 100
EXIT_THRESH = 4000          # wall area on the turning side that ends a turn
CORNER_AREA = {"left": 1000, "right": 1250}     # corner ROI area that forces a sharp turn
TURN_LINE_COOLDOWN = 1.0    # s after a counted turn during which the turn line is ignored
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
REVERSE_COOLDOWN = 1.0
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
    rois = ROIS_WIDE if wide else ROIS_43
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
    corner_on = False

    turn_dir = args.dir or "none"
    t = 0
    l_turn = r_turn = False
    line_cooldown_until = 0.0
    prev_diff = 0
    prev_error = 0
    error = 0
    angle = SERVO_CENTER

    finish_at = None
    last_turn_time = 0.0
    reverse_ready_at = 0.0
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
    print(f"[GO] Driving. Parking after {args.turns} turns. Direction: {turn_dir.upper()}")

    def tick():
        camera.capture_array()   # keep frames fresh during blocking manoeuvres

    def hold(spd, ang, seconds):
        drive.hold(0 if args.steer_only else spd, ang, seconds, tick)

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
            _, close_g = find_pillar(c_green, green_target, "green", pillar, roi_pillar, ctx)
            _, close_r = find_pillar(c_red, red_target, "red", pillar, roi_pillar, ctx)

            if (close_g or close_r) and now >= reverse_ready_at:
                print(f"[PILLAR] Too close (area {max(close_g, close_r)}) -> reversing")
                hold(0, SERVO_CENTER, 0.1)
                hold(REVERSE_SPEED, SERVO_CENTER, 0.5)
                reverse_ready_at = time.time() + REVERSE_COOLDOWN
                continue

            # ---------------------------------------------------------- floor lines / turns
            if turn_dir == "none":
                if orange_area > LINE_MIN_AREA:
                    turn_dir = "right"
                elif blue_area > LINE_MIN_AREA:
                    turn_dir = "left"
                if turn_dir != "none":
                    print(f"[DIR] First line -> direction {turn_dir.upper()}")

            t_signal = False
            if now >= line_cooldown_until and (
                    (turn_dir == "right" and orange_area > LINE_MIN_AREA) or
                    (turn_dir == "left" and blue_area > LINE_MIN_AREA)):
                t_signal = True
                if turn_dir == "right":
                    r_turn = True
                else:
                    l_turn = True
                if pillar.area != 0 and (
                        (left_area > 500 and turn_dir == "left") or (right_area > 500 and turn_dir == "right")):
                    corner_on = True

            def end_turn(method):
                nonlocal l_turn, r_turn, t, prev_error, prev_diff, line_cooldown_until, last_turn_time
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
                    if (l_turn or r_turn) and not t_signal:
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
                if ((r_turn and right_area >= EXIT_THRESH) or (l_turn and left_area >= EXIT_THRESH)) and not t_signal:
                    end_turn("wall")
                if r_turn and pillar.area == 0 and right_area < 5000:
                    angle = SHARP_RIGHT
                elif l_turn and pillar.area == 0 and left_area < 5000:
                    angle = SHARP_LEFT

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
            drive.drive(0 if args.steer_only else speed, angle)

            if link.link_ok != link_was_ok:
                link_was_ok = link.link_ok
                print(f"[LINK] {'restored' if link_was_ok else 'DOWN - car stopped by ESP32 failsafe'}")

            # ---------------------------------------------------------- debug output
            if finish_at is not None:
                state = "FINISHING"
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
