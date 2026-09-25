#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Open Challenge (Round 1) - camera + 6 ultrasonic sensors (hybrid).

This keeps the original ROBOVANGUARD strategy:
  Phase 1  Record a baseline snapshot of the front/back distances in the start section.
  Phase 2  Drive the laps. Steering comes from the black wall areas in the left/right ROIs,
           with the side ultrasonics nudging the car away from a wall it is about to touch.
           The first floor line colour locks the track direction (orange = right, blue = left).
           A corner starts when the inner wall's ROI empties and ends when the camera
           re-acquires a wall (dynamic exit, 0.8-2.2 s).
  Phase 3  After the 12th corner, creep home and stop in the start section, using the
           floor marker, the baseline distances, the front wall and a timeout.
  Any time An obstacle within a few cm triggers a short angled reverse.

Fixed since the version on main:
  - A corner is only counted when the floor marker was seen, or the front sensors confirm
    a wall ahead. Before, the condition collapsed to "a wall area dropped", so any gap in
    the inner wall added a lap.
  - The turn cooldown is no longer overwritten with a shorter value on turn exit.
  - 0 cm now means "nothing in range" instead of "collision", so open track no longer
    triggers emergency reverses. Proximity must also repeat on two updates before it acts.
  - The reverse manoeuvre keeps reading the camera and keeps the ESP32 failsafe fed
    instead of blocking the loop blind for up to 1.2 s.
  - Removed the AUTO_US_ON call before the finish: any DRIVE command cancels it anyway.
  - Any exception now stops the car and prints why, instead of only Ctrl+C.

Usage
  python3 open_challenge_R1.py                  wait for button, run 3 laps
  python3 open_challenge_R1.py --no-display     competition mode
  python3 open_challenge_R1.py --turns 4        test: stop after 1 lap
  python3 open_challenge_R1.py --steer-only     test: motor off, watch steering and sensors
  python3 open_challenge_R1.py --dir right      force direction
  Other: --webcam, --no-wait, --pin N, --active-high
"""

import argparse
import sys
import time
import traceback

import cv2

from wro_serial import WROSerialController, Drive
from wro_functions import (CameraManager, FpsCounter, Ultrasonics, roi_hsv_lab, wall_mask,
                           orange_mask, blue_mask, contours_of, max_contour, draw_roi,
                           draw_offset_contours, wait_for_button_press)

# ============================================================================ tuning
# Camera regions [x1, y1, x2, y2] on the 640x480 frame
ROI_LEFT = [20, 170, 240, 220]
ROI_RIGHT = [400, 170, 620, 220]
ROI_LINE = [200, 300, 440, 350]

# Steering. On this car 60 = full left, 100 = straight, 140 = full right.
SERVO_CENTER = 100
SERVO_MIN, SERVO_MAX = 60, 140
DUAL_WALL_GAIN = 0.015      # both walls visible: centre between them
SINGLE_WALL_GAIN = 0.01     # fallback
CORNER_APPROACH_GAIN = 0.008
CORNER_APPROACH_TARGET = 900
CORNER_APPROACH_CLAMP = (90, 110)
WALL_VISIBLE_AREA = 250     # wall area that counts as "this wall is in view"

# Speed (PWM 0-255)
SPEED = 245
TURN_SPEED = 205
RETURN_SPEED = 230          # controlled speed while creeping to the finish
BRAKE_SPEED = -180
START_DELAY = 0.5

# Turns
TURN_THRESH = 200           # wall area at or below this = that wall has ended
WALL_REACQUIRE_AREA = 600   # wall area that ends the turn
MIN_TURN_TIME = 0.8
MAX_TURN_TIME = 2.2
TURN_COOLDOWN = 1.0         # s after a turn before the next one may start
LINE_MIN_AREA = 100
LINE_LOCKOUT = 1.2          # s after seeing a marker before another counts
POST_TURN_LINE_LOCKOUT = 0.6  # short: the turn cooldown already stops double-counting,
                              # and a long lockout here would swallow the next corner marker
FRONT_CORNER_CM = 45        # a wall this close ahead also confirms a corner
TOTAL_TURNS = 12

# Ultrasonic safety
SIDE_NUDGE_CM = 12          # steer away from a side wall closer than this
FRONT_HIT_CM = 12           # a wall this close ahead triggers reverse
SIDE_HIT_CM = 7
INNER_HIT_CM = 13
REAR_LIMIT_CM = 8
REVERSE_SPEED = -235
REVERSE_MIN_TIME = 0.45
REVERSE_MAX_TIME = 1.2
REVERSE_CLEAR_CM = 18
REVERSE_COOLDOWN = 1.2

# Phase 3: stopping in the start section
MIN_CLEAR_OF_CORNER = 0.5
FINISH_MARKER_AREA = 150
BASELINE_TOLERANCE_CM = 4.0
FRONT_WALL_STOP_CM = 25.0
HOME_TIMEOUT = 4.0

WINDOW = "WRO R1 Open Challenge (hybrid)"


def parse_args():
    p = argparse.ArgumentParser(description="WRO 2026 Open Challenge (camera + ultrasonics)")
    p.add_argument("--no-display", action="store_true", help="no monitor window (competition)")
    p.add_argument("--no-wait", "--nowait", action="store_true", help="start immediately, no button")
    p.add_argument("--webcam", "-w", action="store_true", help="use USB webcam instead of Pi camera")
    p.add_argument("--pin", type=int, default=17, help="start button GPIO (BCM)")
    p.add_argument("--active-high", action="store_true", help="button wired to 3.3 V instead of GND")
    p.add_argument("--dir", choices=["left", "right"], help="force track direction")
    p.add_argument("--turns", type=int, default=TOTAL_TURNS, help="stop after this many turns")
    p.add_argument("--steer-only", action="store_true", help="motor off; steering still reacts")
    p.add_argument("--no-us", "--vision-walls", action="store_true", help="ignore the ultrasonics")
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
    use_us = not args.no_us

    print("=" * 65)
    print("   ROBOVANGUARD - WRO 2026 Open Challenge (camera + 6 ultrasonics)")
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

    camera = CameraManager(force_webcam=args.webcam)
    camera.start()
    for _ in range(15):
        camera.capture_array()

    if not args.no_wait:
        wait_for_button_press(args.pin, args.active_high, camera, show, WINDOW)

    # ---------------------------------------------------------------- Phase 1: baseline
    us.update()
    baseline = dict(us.values)
    print(f"[PHASE 1] Start baseline: {us.text()}")
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
    link_was_ok = True
    exit_reason = "unknown"

    fps = FpsCounter()
    t_start = time.time()
    last_status = 0.0
    print(f"[PHASE 2] Driving. Target {args.turns} turns. Direction: {turn_dir.upper()}")

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
            c_left = contours_of(wall_mask(hsv, lab), 50)
            hsv, lab = roi_hsv_lab(img, ROI_RIGHT)
            c_right = contours_of(wall_mask(hsv, lab), 50)
            hsv, lab = roi_hsv_lab(img, ROI_LINE)
            c_orange = contours_of(orange_mask(hsv, lab), LINE_MIN_AREA)
            c_blue = contours_of(blue_mask(hsv, lab), LINE_MIN_AREA)

            left_area = max_contour(c_left, ROI_LEFT)[0]
            right_area = max_contour(c_right, ROI_RIGHT)[0]
            orange_area = max_contour(c_orange, ROI_LINE)[0]
            blue_area = max_contour(c_blue, ROI_LINE)[0]

            # ---------------------------------------------------------- emergency reverse
            if use_us and not returning_home and not is_turning and now >= reverse_ready_at:
                right_hit = us.near("r", SIDE_HIT_CM) or (turn_dir == "right" and us.near("f2", INNER_HIT_CM))
                left_hit = us.near("l", SIDE_HIT_CM) or (turn_dir == "left" and us.near("f1", INNER_HIT_CM))
                front_hit = us.near("f", FRONT_HIT_CM) or (us.near("f1", FRONT_HIT_CM) and us.near("f2", FRONT_HIT_CM))

                if right_hit or left_hit or front_hit:
                    if right_hit:
                        rev_steer, what = 65, "RIGHT/INNER WALL"
                    elif left_hit:
                        rev_steer, what = 135, "LEFT/INNER WALL"
                    else:
                        rev_steer, what = (65 if turn_dir == "right" else 135), "FRONT WALL"
                    print(f"[REVERSE] {what} too close ({us.text()}) -> backing off at {rev_steer} deg")

                    rev_start = time.time()
                    while time.time() - rev_start < REVERSE_MAX_TIME:
                        camera.capture_array()          # keep the camera stream fresh
                        us.update()
                        drive.drive(0 if args.steer_only else REVERSE_SPEED, rev_steer)
                        if us.near("b", REAR_LIMIT_CM):
                            print(f"[REVERSE] Rear wall at {us.get('b')} cm - stopping reverse")
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
                        line_lockout_until = now + LINE_LOCKOUT
                        print(f"[DIR] First line ORANGE ({orange_area} px) -> direction RIGHT (locked)")
                    elif blue_area > LINE_MIN_AREA:
                        turn_dir, marker_seen = "left", True
                        line_lockout_until = now + LINE_LOCKOUT
                        print(f"[DIR] First line BLUE ({blue_area} px) -> direction LEFT (locked)")
                elif ((turn_dir == "right" and orange_area > LINE_MIN_AREA) or
                      (turn_dir == "left" and blue_area > LINE_MIN_AREA)):
                    marker_seen = True
                    line_lockout_until = now + LINE_LOCKOUT

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
                    if turns >= args.turns:
                        returning_home = True
                        corner12_time = now
                        print(f"[PHASE 3] Last corner cleared - creeping to the finish at {RETURN_SPEED}")
                else:
                    drive.drive(0 if args.steer_only else TURN_SPEED, target)

            elif not returning_home and turns < args.turns and now >= cooldown_until:
                wall_dropped = ((left_area <= TURN_THRESH and right_area <= TURN_THRESH)
                                or (turn_dir == "left" and left_area <= TURN_THRESH)
                                or (turn_dir == "right" and right_area <= TURN_THRESH))
                # A corner needs the wall to end AND a confirmation: the floor marker, or
                # (with the sensors) a wall close ahead. This is the fix for phantom laps.
                confirmed = marker_seen or (use_us and us.near("f", FRONT_CORNER_CM))
                if wall_dropped and confirmed and turn_dir != "none":
                    is_turning = True
                    turn_start = now
                    target = SERVO_MIN if turn_dir == "left" else SERVO_MAX
                    why = "marker" if marker_seen else "front wall"
                    print(f"[TURN] Corner {turns + 1} starting ({why}; L={left_area} R={right_area})")
                    drive.drive(0 if args.steer_only else TURN_SPEED, target)

            # ---------------------------------------------------------- straight steering
            if not is_turning:
                both_walls = left_area > WALL_VISIBLE_AREA and right_area > WALL_VISIBLE_AREA
                if both_walls:
                    angle = SERVO_CENTER - (right_area - left_area) * DUAL_WALL_GAIN
                elif turn_dir == "right" and right_area <= WALL_VISIBLE_AREA:
                    # inner wall gone at the corner: hold course on the outer wall,
                    # do not steer into the inside of the corner
                    angle = SERVO_CENTER - (left_area - CORNER_APPROACH_TARGET) * CORNER_APPROACH_GAIN
                    angle = clamp(angle, *CORNER_APPROACH_CLAMP)
                elif turn_dir == "left" and left_area <= WALL_VISIBLE_AREA:
                    angle = SERVO_CENTER + (right_area - CORNER_APPROACH_TARGET) * CORNER_APPROACH_GAIN
                    angle = clamp(angle, *CORNER_APPROACH_CLAMP)
                else:
                    angle = SERVO_CENTER - (right_area - left_area) * SINGLE_WALL_GAIN

                if use_us:      # nudge away from a side wall we are about to touch
                    if us.near("r", SIDE_NUDGE_CM):
                        angle = min(angle, SERVO_CENTER - 20)
                    elif us.near("l", SIDE_NUDGE_CM):
                        angle = max(angle, SERVO_CENTER + 20)

                angle = int(clamp(angle, SERVO_MIN, SERVO_MAX))

                if returning_home:
                    # ------------------------------------------------ Phase 3: stop at home
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
                    drive.drive(0 if args.steer_only else RETURN_SPEED, angle)
                else:
                    deflection = abs(angle - SERVO_CENTER)
                    speed = TURN_SPEED if deflection > 30 else SPEED
                    drive.drive(0 if args.steer_only else speed, angle)
            else:
                angle = SERVO_MIN if turn_dir == "left" else SERVO_MAX

            if link.link_ok != link_was_ok:
                link_was_ok = link.link_ok
                print(f"[LINK] {'restored' if link_was_ok else 'DOWN - car stopped by ESP32 failsafe'}")

            # ---------------------------------------------------------- debug output
            state = ("RETURN_HOME" if returning_home else
                     f"TURN-{turn_dir[0].upper()}" if is_turning else "STRAIGHT")
            if status_to_terminal and now - last_status > 0.2:
                last_status = now
                print(f"\r{state:11s} t={turns:2d} L={left_area:5d} R={right_area:5d} "
                      f"ang={angle:3d} {us.text()} fps={fps.fps:4.1f} "
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
