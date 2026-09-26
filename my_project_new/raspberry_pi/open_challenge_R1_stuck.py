#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Open Challenge (Round 1) - camera only, WITH STUCK RECOVERY.

This is open_challenge_R1.py plus one extra behaviour: if the car wedges itself against
a wall, it backs out and carries on instead of sitting there grinding the wheels.

How being stuck is detected
  NOT by comparing frames for exact equality - sensor noise means two frames are almost
  never identical, even with the car held still, so that test would never fire. Instead
  it measures HOW MUCH the picture changes between frames (mean absolute difference of
  small greyscale copies). When that stays below STUCK_STILL_THRESHOLD for
  STUCK_TIME seconds WHILE the motor is being commanded to drive, the car is stuck.
  The live number is printed as mot= in the status line, so you can set the threshold
  from what your own camera reports.

How it gets out
  It reverses with the steering turned toward the side it is jammed against, which
  swings the nose away from that side. Reversing with the wheels turned the other way
  would push the nose harder into the wall. The side is taken from the turn the car was
  making, or from which wall ROI is fuller when driving straight - so the direction of
  the turns is what decides the escape, as you asked.
  If it gets stuck again straight afterwards, it alternates the side and reverses for
  longer, in case the first guess was wrong.

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
  Stuck    : view stops changing while driving -> reverse out, then resume.

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
                           wait_for_button_press, roi_px, area_norm, is_wide, apply_roi_config,
                           StuckDetector)

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

# Stuck detection and recovery
STUCK_STILL_THRESHOLD = 1.5   # picture-change score below this counts as "not moving".
                              # Watch mot= in the status line: near 0 when held still,
                              # several times higher when driving. Put this in between.
STUCK_TIME = 1.2              # s of no change, while driving, before calling it stuck
REVERSE_SPEED = -215          # PWM while backing out (negative = reverse)
REVERSE_MIN_TIME = 0.35       # always reverse at least this long
REVERSE_MAX_TIME = 1.1        # give up reversing after this
REVERSE_EXTRA_PER_TRY = 0.35  # each repeat attempt reverses this much longer
FREED_MOTION = 3.0            # picture-change score that means we are moving again
RECOVER_COOLDOWN = 1.0        # s after a recovery before another can trigger
REPEAT_WINDOW = 6.0           # repeats within this long count as the same jam
MAX_RECOVERIES = 6            # consecutive failed attempts before giving up and stopping

WINDOW = "WRO R1 Open Challenge (stuck recovery)"


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
    p.add_argument("--no-stuck-recovery", action="store_true", help="detect stuck but never reverse")
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
    stuck = StuckDetector(still_threshold=STUCK_STILL_THRESHOLD, stuck_time=STUCK_TIME)
    turn_history = []            # every turn the car has made, newest last
    recoveries = 0               # total this run
    repeat_tries = 0             # consecutive attempts on what looks like the same jam
    last_recovery_end = 0.0
    flip_side = False
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
                    side = "right" if r_turn else "left"
                    l_turn = r_turn = False
                    prev_diff = 0
                    cooldown_until = now + TURN_COOLDOWN
                    last_turn_time = now
                    turn_history.append(side)
                    if l_detected:
                        turns += 1
                        print(f"[TURN] {turns}/{args.turns} {side.upper()} done at {now - t_start:.1f}s")
                    else:
                        print(f"[TURN] {side.upper()} ended without a floor line -> not counted")
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

            # ---------------------------------------------------------- stuck recovery
            driving = speed != 0 and link.link_ok
            is_stuck = stuck.update(img, moving=driving, now=now)

            if is_stuck and not args.no_stuck_recovery and now >= last_recovery_end + RECOVER_COOLDOWN:
                # Same jam as last time, or a fresh one?
                if now - last_recovery_end <= REPEAT_WINDOW:
                    repeat_tries += 1
                    flip_side = not flip_side      # first guess did not work: try the other way
                else:
                    repeat_tries = 1
                    flip_side = False
                recoveries += 1

                # Which side are we jammed against? The turn we are making, or the fuller
                # wall when going straight. Reversing with the wheels toward that side
                # swings the nose away from it.
                if l_turn or turn_dir == "left":
                    side = "left"
                elif r_turn or turn_dir == "right":
                    side = "right"
                else:
                    side = "left" if left_area >= right_area else "right"
                if flip_side:
                    side = "right" if side == "left" else "left"
                rev_angle = SERVO_MIN if side == "left" else SERVO_MAX

                if repeat_tries > MAX_RECOVERIES:
                    print(f"\n[STUCK] Still jammed after {MAX_RECOVERIES} attempts - stopping.")
                    drive.stop(0 if args.steer_only else BRAKE_SPEED, SERVO_CENTER)
                    exit_reason = f"stuck after {recoveries} recovery attempts"
                    break

                limit = min(REVERSE_MAX_TIME + REVERSE_EXTRA_PER_TRY * (repeat_tries - 1), 2.0)
                print(f"\n[STUCK] No movement for {STUCK_TIME:.1f}s (mot={stuck.motion:.2f}) "
                      f"after turn {turns}. Backing out to the {side} for up to {limit:.2f}s "
                      f"(attempt {repeat_tries})")

                drive.drive(0, SERVO_CENTER, force=True)
                rev_start = time.time()
                free_frames = 0
                while time.time() - rev_start < limit:
                    f = camera.capture_array()
                    if f is None:
                        continue
                    drive.drive(0 if args.steer_only else REVERSE_SPEED, rev_angle)
                    stuck.update(f, moving=True)
                    free_frames = free_frames + 1 if stuck.motion > FREED_MOTION else 0
                    if show:
                        cv2.imshow(WINDOW, f)
                        cv2.waitKey(1)
                    if time.time() - rev_start >= REVERSE_MIN_TIME and free_frames >= 3:
                        break

                drive.drive(0, SERVO_CENTER, force=True)
                freed_motion = stuck.motion
                stuck.reset()
                last_recovery_end = time.time()
                # Do not let the changed view trigger a phantom corner straight afterwards
                cooldown_until = max(cooldown_until, last_recovery_end + TURN_COOLDOWN)
                got_free = "moving again" if freed_motion > FREED_MOTION else "still not moving"
                print(f"[STUCK] Resuming - {got_free} (mot={freed_motion:.2f}, "
                      f"recoveries this run: {recoveries})")
                continue

            if link.link_ok != link_was_ok:
                link_was_ok = link.link_ok
                print(f"[LINK] {'restored' if link_was_ok else 'DOWN - car stopped by ESP32 failsafe'}")

            # ---------------------------------------------------------- debug output
            state = "TURN-L" if l_turn else "TURN-R" if r_turn else "STRAIGHT"
            if status_to_terminal and now - last_status > 0.2:
                last_status = now
                print(f"\r{state:8s} t={turns:2d} L={left_area:5d} R={right_area:5d} "
                      f"O={orange_area:4d} B={blue_area:4d} ang={angle:3d} mot={stuck.motion:5.2f} "
                      f"rec={recoveries} fps={fps.fps:4.1f} "
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
        if recoveries:
            print(f"[EXIT] Stuck recoveries this run: {recoveries} "
                  f"(turns made: {', '.join(turn_history) if turn_history else 'none'})")
        print(f"[EXIT] Run ended: {exit_reason} | turns {turns}/{args.turns} | "
              f"{time.time() - t_start:.1f}s | avg fps {fps.fps:.1f}")
        link.disconnect()


if __name__ == "__main__":
    main()
