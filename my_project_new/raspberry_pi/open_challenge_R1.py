#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Open Challenge (Round 1) - camera only.

Strategy follows the Canada team (canada-team/src/OpenChallenge.py), adapted to this car:
  Straight : PD steering on (right wall area - left wall area) measured in two fixed ROIs.
             A closer wall fills more of its ROI, so equal areas = centred in the lane.
  Turn in  : the inner wall ends at a corner, so its ROI empties (area <= TURN_THRESH).
             Steer at least TURN_DEVIATION toward that side.
  Turn out : the same ROI fills again with the next wall (area > EXIT_THRESH).
  Counting : a finished turn only counts if the turn-colour floor line was seen, so a
             wall gap on a straight can never add a lap.
  Finish   : after 12 counted turns, wait until steering is straight, drive on for
             FINISH_DELAY seconds, then brake and stop in the start section.

Changes vs Canada's code: fixes their blue-line typo ('lDeteted'), locks turns to the
detected direction, stops without a blocking sleep (a blocking sleep would trigger the
ESP32 500 ms failsafe), converts only the ROIs instead of the whole frame.

Usage
  python3 open_challenge_R1.py                  wait for button, run 3 laps, with monitor window
  python3 open_challenge_R1.py --no-display     competition mode
  python3 open_challenge_R1.py --turns 4        test: stop after 1 lap
  python3 open_challenge_R1.py --steer-only     test: motor off, watch the steering react
  python3 open_challenge_R1.py --dir right      force the direction instead of detecting it
  Other: --webcam, --no-wait, --pin N, --active-high
"""

import argparse
import sys
import time
import traceback

import cv2

from wro_serial import WROSerialController, Drive
from wro_functions import (CameraManager, FpsCounter, roi_hsv_lab, wall_mask, orange_mask, blue_mask,
                           contours_of, max_contour, draw_roi, draw_offset_contours,
                           wait_for_button_press, roi_px, area_norm, is_wide, apply_roi_config)

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
# Re-check these whenever the camera mount changes: the side walls must fill most of the
# left/right ROIs on a straight.
ROIS_43 = {
    "left":  (0.031, 0.354, 0.375, 0.458),
    "right": (0.625, 0.354, 0.969, 0.458),
    "line":  (0.313, 0.625, 0.688, 0.729),
}
# Wide defaults sit further out and are taller than the 4:3 ones: a wide lens puts the
# side walls near the edges of the frame. These are a starting point only - set them on
# the real track with:  python3 tune_rois.py
ROIS_WIDE = {
    "left":  (0.02, 0.42, 0.32, 0.58),
    "right": (0.68, 0.42, 0.98, 0.58),
    "line":  (0.33, 0.70, 0.67, 0.86),
}

# Steering. On this car 60 = full left, 100 = straight, 140 = full right.
SERVO_CENTER = 100
# Usable steering range: 100 = straight, 70 = full left, 130 = full right (+-30).
# Past about 30 deg the front tyres scrub and skid instead of steering, so this is the
# usable limit rather than the mechanical one.
SERVO_MIN, SERVO_MAX = 70, 130
KP = 0.02               # Canada: 0.02
KD = 0.006              # Canada: 0.006
STRAIGHT_LIMIT = 18     # max steering offset from centre on straights
TURN_DEVIATION = 30     # steering offset in a corner. 30 = full lock, the angle where
                        # the tyres still grip. Lower it if the car cuts corners.

# Turns
TURN_THRESH = 150       # wall area at or below this: that wall has ended, start turning
EXIT_THRESH = 1500      # wall area above this on the turning side: turn finished
WALL_MIN_AREA = 50      # ignore wall blobs smaller than this
LINE_MIN_AREA = 100     # a floor-line blob larger than this counts as "line seen"
TURN_COOLDOWN = 0.5     # s after a turn ends before a new one may start
TOTAL_TURNS = 12

# Speed (PWM 0-255). Start conservative, raise once the turns are reliable.
SPEED = 230
TURN_SPEED = 210
START_DELAY = 0.5       # s between the button press and moving

# Finish: after the last turn, drive on this long once straight, then stop.
FINISH_DELAY = {"left": 1.0, "right": 1.5, "none": 1.25}   # Canada: 1.0 / 1.5 s
FINISH_STRAIGHT_TOL = 10    # steering must be within this of centre to start the finish timer
FINISH_MAX_WAIT = 3.0       # if it never settles that straight, stop anyway this long after the last turn
BRAKE_SPEED = -180          # short reverse pulse to stop quickly (0 = coast to a stop)

WINDOW = "WRO R1 Open Challenge"


def parse_args():
    p = argparse.ArgumentParser(description="WRO 2026 Open Challenge (camera only)")
    p.add_argument("--no-display", action="store_true", help="no monitor window (competition)")
    p.add_argument("--no-wait", "--nowait", action="store_true", help="start immediately, no button")
    p.add_argument("--webcam", "-w", action="store_true", help="use USB webcam instead of Pi camera")
    p.add_argument("--pin", type=int, default=17, help="start button GPIO (BCM)")
    p.add_argument("--active-high", action="store_true", help="button wired to 3.3 V instead of GND")
    p.add_argument("--dir", choices=["left", "right"], help="force track direction")
    p.add_argument("--turns", type=int, default=TOTAL_TURNS, help="stop after this many turns")
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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    args = parse_args()
    show = not args.no_display
    status_to_terminal = sys.stdout.isatty()

    print("=" * 65)
    print("   ROBOVANGUARD - WRO 2026 Open Challenge (camera only, Canada strategy)")
    print("=" * 65)

    link = WROSerialController(auto_connect=False)
    link.connect(wait=5.0)          # if not found yet, it keeps retrying in the background
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
    for _ in range(15):             # let auto-exposure settle, and learn the frame size
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
    ROI_LEFT = roi_px(rois["left"], FRAME_W, FRAME_H)
    ROI_RIGHT = roi_px(rois["right"], FRAME_W, FRAME_H)
    ROI_LINE = roi_px(rois["line"], FRAME_W, FRAME_H)
    AREA = area_norm(FRAME_W, FRAME_H)
    print(f"[CAMERA] {FRAME_W}x{FRAME_H} "
          f"({'16:9 - full sensor width' if wide else '4:3 - sides cropped off the sensor'})")
    print(f"[ROI] left {ROI_LEFT}  right {ROI_RIGHT}  line {ROI_LINE}")

    if not args.no_wait:
        wait_for_button_press(args.pin, args.active_high, camera, show, WINDOW)

    warn = link.health_warning()
    if warn:
        print("=" * 66)
        print(f"[WARNING] Before starting: {warn}")
        print("=" * 66)
    time.sleep(START_DELAY)

    # ------------------------------------------------------------------ run state
    turn_dir = args.dir or "none"
    l_turn = r_turn = False
    l_detected = False          # a floor line was seen since the last counted turn
    turns = 0
    prev_diff = 0
    cooldown_until = 0.0
    finish_at = None
    last_turn_time = 0.0
    link_was_ok = True
    exit_reason = "unknown"

    fps = FpsCounter()
    t_start = time.time()
    last_status = 0.0
    print(f"[GO] Driving. Target {args.turns} turns. Direction: {turn_dir.upper()}")

    try:
        while True:
            img = camera.capture_array()
            if img is None:
                time.sleep(0.005)    # no frame: send nothing, the ESP32 failsafe stops the car
                continue
            now = time.time()
            fps.tick()

            # ---------------------------------------------------------- vision
            hsv, lab = roi_hsv_lab(img, ROI_LEFT)
            c_left = contours_of(wall_mask(hsv, lab), WALL_MIN_AREA)
            hsv, lab = roi_hsv_lab(img, ROI_RIGHT)
            c_right = contours_of(wall_mask(hsv, lab), WALL_MIN_AREA)
            hsv, lab = roi_hsv_lab(img, ROI_LINE)
            c_orange = contours_of(orange_mask(hsv, lab), LINE_MIN_AREA)
            c_blue = contours_of(blue_mask(hsv, lab), LINE_MIN_AREA)

            # areas normalised to 640x480-equivalent pixels so the thresholds above hold
            left_area = int(max_contour(c_left, ROI_LEFT)[0] * AREA)
            right_area = int(max_contour(c_right, ROI_RIGHT)[0] * AREA)
            orange_area = int(max_contour(c_orange, ROI_LINE)[0] * AREA)
            blue_area = int(max_contour(c_blue, ROI_LINE)[0] * AREA)

            # ---------------------------------------------------------- floor lines
            orange_seen = orange_area > LINE_MIN_AREA
            blue_seen = blue_area > LINE_MIN_AREA
            if turn_dir == "none" and (orange_seen or blue_seen):
                turn_dir = "right" if orange_area >= blue_area else "left"
                print(f"[DIR] First line {'ORANGE' if turn_dir == 'right' else 'BLUE'} -> direction {turn_dir.upper()}")
            # Only the turn-colour line marks a corner entry (orange for right, blue for left)
            if (turn_dir == "right" and orange_seen) or (turn_dir == "left" and blue_seen):
                l_detected = True

            # ---------------------------------------------------------- PD steering
            a_diff = right_area - left_area
            angle = SERVO_CENTER - (a_diff * KP + (a_diff - prev_diff) * KD)
            prev_diff = a_diff

            # ---------------------------------------------------------- turn state
            if not (l_turn or r_turn) and now >= cooldown_until and turns < args.turns:
                if left_area <= TURN_THRESH and turn_dir in ("none", "left"):
                    l_turn = True
                elif right_area <= TURN_THRESH and turn_dir in ("none", "right"):
                    r_turn = True

            if l_turn or r_turn:
                if (r_turn and right_area > EXIT_THRESH) or (l_turn and left_area > EXIT_THRESH):
                    side = "RIGHT" if r_turn else "LEFT"
                    l_turn = r_turn = False
                    prev_diff = 0
                    cooldown_until = now + TURN_COOLDOWN
                    last_turn_time = now
                    if l_detected:
                        turns += 1
                        print(f"[TURN] {turns}/{args.turns} {side} done at {now - t_start:.1f}s")
                    else:
                        print(f"[TURN] {side} ended without a floor line -> not counted")
                    l_detected = False
                    angle = clamp(angle, SERVO_CENTER - STRAIGHT_LIMIT, SERVO_CENTER + STRAIGHT_LIMIT)
                elif l_turn:
                    angle = clamp(min(angle, SERVO_CENTER - TURN_DEVIATION), SERVO_MIN, SERVO_MAX)
                else:
                    angle = clamp(max(angle, SERVO_CENTER + TURN_DEVIATION), SERVO_MIN, SERVO_MAX)
            else:
                angle = clamp(angle, SERVO_CENTER - STRAIGHT_LIMIT, SERVO_CENTER + STRAIGHT_LIMIT)
            angle = int(angle)

            # ---------------------------------------------------------- finish
            if turns >= args.turns and finish_at is None:
                if not (l_turn or r_turn) and abs(angle - SERVO_CENTER) <= FINISH_STRAIGHT_TOL:
                    finish_at = now + FINISH_DELAY[turn_dir]
                    print(f"[FINISH] All turns done. Stopping in {FINISH_DELAY[turn_dir]:.2f}s")
                elif now - last_turn_time > FINISH_MAX_WAIT:
                    finish_at = now      # steering never settled straight: stop anyway
                    print("[FINISH] Steering never settled straight - stopping now")
            if finish_at is not None and now >= finish_at:
                drive.stop(0 if args.steer_only else BRAKE_SPEED, SERVO_CENTER)
                exit_reason = f"finished {turns} turns"
                break

            # ---------------------------------------------------------- drive
            if args.steer_only:
                speed = 0
            else:
                speed = TURN_SPEED if (l_turn or r_turn) else SPEED
            drive.drive(speed, angle)

            if link.link_ok != link_was_ok:
                link_was_ok = link.link_ok
                print(f"[LINK] {'restored' if link_was_ok else 'DOWN - car stopped by ESP32 failsafe'}")

            # ---------------------------------------------------------- debug output
            state = "TURN-L" if l_turn else "TURN-R" if r_turn else "STRAIGHT"
            if status_to_terminal and now - last_status > 0.2:
                last_status = now
                print(f"\r{state:8s} t={turns:2d} L={left_area:5d} R={right_area:5d} "
                      f"O={orange_area:4d} B={blue_area:4d} ang={angle:3d} fps={fps.fps:4.1f} "
                      f"link={'ok' if link.link_ok else 'DOWN'}   ", end="", flush=True)

            if show:
                disp = img.copy()
                draw_roi(disp, ROI_LEFT, (0, 255, 255))
                draw_roi(disp, ROI_RIGHT, (0, 255, 255))
                draw_roi(disp, ROI_LINE, (255, 255, 0))
                draw_offset_contours(disp, c_left, ROI_LEFT, (0, 255, 0))
                draw_offset_contours(disp, c_right, ROI_RIGHT, (0, 255, 0))
                draw_offset_contours(disp, c_orange, ROI_LINE, (0, 165, 255))
                draw_offset_contours(disp, c_blue, ROI_LINE, (255, 0, 0))
                cv2.putText(disp, f"{state} dir:{turn_dir} turns:{turns}/{args.turns} fps:{fps.fps:.0f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 204), 2)
                cv2.putText(disp, f"L:{left_area} R:{right_area} O:{orange_area} B:{blue_area} ang:{angle}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(disp, "link OK" if link.link_ok else "LINK DOWN", (10, 75),
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
