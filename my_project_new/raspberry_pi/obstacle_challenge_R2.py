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
  Parking  After 12 corners, look for the magenta lot and drive in, stopping on the
           front sensor.
  Any time An obstacle within a few cm triggers a short reverse.

Fixed since the version on main:
  - endConst is now actually used. It was computed every frame and never applied, so a
    pillar kept steering the car after it had passed the nose - which is how a rear wheel
    clips a block and loses the 10-point "no signs moved" bonus.
  - A corner is only counted with the floor marker or a front-sensor confirmation.
  - 0 cm now means "nothing in range" instead of "collision", and proximity must repeat
    on two updates before it acts.
  - Reverse and parking no longer block the loop blind; they keep reading the camera and
    keep the ESP32 failsafe fed.
  - Parking stops on the front sensor instead of driving blind for a fixed 2 s.
  - If the lot is never found, the car stops instead of driving until the round times out.
  - Any exception now stops the car and prints why, instead of only Ctrl+C.

Usage
  python3 obstacle_challenge_R2.py                 wait for button, 3 laps + parking
  python3 obstacle_challenge_R2.py --no-display    competition mode
  python3 obstacle_challenge_R2.py --turns 4       test: park after 1 lap
  python3 obstacle_challenge_R2.py --parking       test: start the parking search now
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
                           wait_for_button_press)

# ============================================================================ tuning
RED_TARGET = 110            # image x a red pillar is steered to (pass it on the right)
GREEN_TARGET = 530          # image x a green pillar is steered to (pass it on the left)

# Camera regions [x1, y1, x2, y2] on the 640x480 frame
ROI_LEFT = [20, 170, 240, 220]
ROI_RIGHT = [400, 170, 620, 220]
ROI_PILLAR = [0, 60, 640, 280]
ROI_FLOOR = [200, 270, 440, 340]

# Steering. On this car 60 = full left, 100 = straight, 140 = full right.
SERVO_CENTER = 100
SERVO_MIN, SERVO_MAX = 60, 140

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
PARK_SPEED = 225
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

# Parking
PARK_MAGENTA_AREA = 1800    # magenta this big in the floor ROI = the lot is in front
PARK_FRONT_STOP_CM = 15     # stop once the barrier is this close
PARK_MAX_TIME = 3.0         # s of driving into the bay before stopping anyway
PARK_SEARCH_TIMEOUT = 20.0  # s after the last corner without finding the lot -> stop
PARK_REVERSE_CM = 8         # reverse threshold while parking (must be < PARK_FRONT_STOP_CM)

WINDOW = "WRO R2 Obstacle Challenge (hybrid)"


class Pillar:
    def __init__(self):
        self.area = 0
        self.dist = 1_000_000
        self.x = 0
        self.y = 0
        self.target = GREEN_TARGET
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
    p.add_argument("--parking", action="store_true", help="test: start the parking search immediately")
    p.add_argument("--no-us", action="store_true", help="ignore the ultrasonics")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[CONFIG] Ignoring old/unknown arguments: {unknown}")
    return args


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def find_pillar(contours, target, colour, best, parking, end_const):
    """
    Picks the nearest usable pillar of one colour. Pillars that have slid below
    (ROI bottom - end_const) are dropped, so a block the car has already passed stops
    pulling the steering - that drop was missing on main.
    """
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < PILLAR_MIN_AREA:
            continue
        if parking and colour == "green" and area < 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x += ROI_PILLAR[0] + w // 2
        y += ROI_PILLAR[1] + h
        dist = round(math.dist([x, y], [320, 480]))

        if 80 < dist < PILLAR_MAX_DIST:
            count += 1
        if dist > PILLAR_MAX_DIST:
            continue
        if y > ROI_PILLAR[3] - end_const:     # passing under the nose: stop tracking it
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

    camera = CameraManager(force_webcam=args.webcam)
    camera.start()
    for _ in range(15):
        camera.capture_array()

    if not args.no_wait:
        wait_for_button_press(args.pin, args.active_high, camera, show, WINDOW)
    time.sleep(START_DELAY)

    # ---------------------------------------------------------------- run state
    turn_dir = args.dir or "none"
    turns = args.turns if args.parking else 0
    is_turning = False
    turn_start = 0.0
    marker_seen = False
    line_lockout_until = 0.0
    cooldown_until = 0.0
    reverse_ready_at = 0.0
    searching_lot = args.parking
    lot_search_start = time.time() if args.parking else 0.0
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
            c_left = contours_of(wall_mask(hsv, lab), 50)
            hsv, lab = roi_hsv_lab(img, ROI_RIGHT)
            c_right = contours_of(wall_mask(hsv, lab), 50)
            hsv_f, lab_f = roi_hsv_lab(img, ROI_FLOOR)
            c_orange = contours_of(orange_mask(hsv_f, lab_f), LINE_MIN_AREA)
            c_blue = contours_of(blue_mask(hsv_f, lab_f), LINE_MIN_AREA)
            c_magenta = contours_of(magenta_mask(lab_f), 100)
            hsv_p, _ = roi_hsv_lab(img, ROI_PILLAR)
            c_red = contours_of(red_mask(hsv_p), PILLAR_MIN_AREA - 1, PILLAR_RED_MIN_ASPECT)
            c_green = contours_of(green_mask(hsv_p), PILLAR_MIN_AREA - 1, PILLAR_GREEN_MIN_ASPECT)

            left_area = max_contour(c_left, ROI_LEFT)[0]
            right_area = max_contour(c_right, ROI_RIGHT)[0]
            orange_area = max_contour(c_orange, ROI_FLOOR)[0]
            blue_area = max_contour(c_blue, ROI_FLOOR)[0]
            magenta = max_contour(c_magenta, ROI_FLOOR)

            # ---------------------------------------------------------- pillars
            probe = Pillar()
            n_g = find_pillar(c_green, GREEN_TARGET, "green", probe, searching_lot, 20)
            n_r = find_pillar(c_red, RED_TARGET, "red", probe, searching_lot, 20)
            gains = "crowded" if (n_g >= 2 or n_r >= 2) else "normal"
            c_kp, c_kd, c_y, end_const = PILLAR_GAINS[gains]

            pillar = Pillar()
            find_pillar(c_green, GREEN_TARGET, "green", pillar, searching_lot, end_const)
            find_pillar(c_red, RED_TARGET, "red", pillar, searching_lot, end_const)

            # ---------------------------------------------------------- emergency reverse
            if use_us and not is_turning and now >= reverse_ready_at:
                # While hunting the lot the car drives at the barrier on purpose, so the
                # reverse threshold drops below PARK_FRONT_STOP_CM and parking wins.
                hit_cm = PARK_REVERSE_CM if searching_lot else FRONT_HIT_CM
                front_hit = (us.near("f", hit_cm) or us.near("f1", hit_cm)
                             or us.near("f2", hit_cm))
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
            if not searching_lot and now >= line_lockout_until:
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
                    if turns >= args.turns and not searching_lot:
                        searching_lot = True
                        lot_search_start = now
                        print("[PARKING] Laps complete -> searching for the magenta parking lot")
                else:
                    angle = target
                    drive.drive(0 if args.steer_only else TURN_SPEED, target)

            elif not searching_lot and turns < args.turns and now >= cooldown_until:
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

            # ---------------------------------------------------------- parking
            if searching_lot and not is_turning:
                if magenta[0] > PARK_MAGENTA_AREA:
                    midpoint = ROI_FLOOR[0] + (ROI_FLOOR[2] - ROI_FLOOR[0]) // 2
                    park_angle = SERVO_MIN if magenta[1] < midpoint else SERVO_MAX
                    side = "LEFT" if park_angle == SERVO_MIN else "RIGHT"
                    print(f"[PARKING] Lot found (area {magenta[0]}, x {magenta[1]}) -> turning in to the {side}")

                    park_start = time.time()
                    while time.time() - park_start < PARK_MAX_TIME:
                        camera.capture_array()
                        us.update()
                        drive.drive(0 if args.steer_only else PARK_SPEED, park_angle)
                        if use_us and us.near("f", PARK_FRONT_STOP_CM):
                            print(f"[PARKING] Barrier at {us.get('f')} cm - stopping")
                            break
                        if show:
                            cv2.waitKey(1)
                        time.sleep(0.01)

                    drive.stop(0 if args.steer_only else BRAKE_SPEED, SERVO_CENTER)
                    exit_reason = "parked"
                    break

                if now - lot_search_start > PARK_SEARCH_TIMEOUT:
                    drive.stop(0 if args.steer_only else BRAKE_SPEED, SERVO_CENTER)
                    exit_reason = "parking lot not found within the timeout"
                    break

            # ---------------------------------------------------------- steering
            if not is_turning:
                if pillar.area > 0 or (now < evade_until and evade_target is not None):
                    if pillar.area > 0:
                        evade_target = pillar.target
                        evade_until = now + EVADE_HOLD
                        error = pillar.target - pillar.x
                        angle = SERVO_CENTER - (error * c_kp) - ((error - prev_error) * c_kd)
                        if not searching_lot:
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
                    speed = PARK_SPEED if searching_lot else SPEED

                angle = int(clamp(angle, SERVO_MIN, SERVO_MAX))
                drive.drive(0 if args.steer_only else speed, angle)

            if link.link_ok != link_was_ok:
                link_was_ok = link.link_ok
                print(f"[LINK] {'restored' if link_was_ok else 'DOWN - car stopped by ESP32 failsafe'}")

            # ---------------------------------------------------------- debug output
            if is_turning:
                state = f"TURN-{turn_dir[0].upper()}"
            elif searching_lot:
                state = "SEARCH-LOT"
            elif pillar.area:
                state = "PILLAR-R" if pillar.target == RED_TARGET else "PILLAR-G"
            elif now < evade_until:
                state = "EVADING"
            else:
                state = "WALLS"

            if status_to_terminal and now - last_status > 0.2:
                last_status = now
                print(f"\r{state:10s} t={turns:2d} L={left_area:5d} R={right_area:5d} "
                      f"P={int(pillar.area):5d} M={magenta[0]:5d} ang={angle:3d} {us.text()} "
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
                draw_offset_contours(disp, c_magenta, ROI_FLOOR, (255, 0, 255))
                if pillar.area:
                    cv2.circle(disp, (int(pillar.x), int(pillar.y)), 6, (255, 255, 255), -1)
                    cv2.line(disp, (pillar.target, 0), (pillar.target, 479), (255, 255, 255), 1)
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
