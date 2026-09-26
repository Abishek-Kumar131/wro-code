#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Interactive ROI tuner.

Why this exists
  Every ROI box started life on another team's camera, at another mount height, with
  another lens. On your car they point at the wrong part of the world, and every
  threshold downstream is measured through them. This sets them on your actual track,
  in a live view, and saves the result so both challenge scripts pick it up.

  Boxes are stored as fractions of the frame, so they survive a resolution change.

Usage
  Put the car on the track where it starts, then:

      python3 tune_rois.py              # open challenge boxes (left, right, line)
      python3 tune_rois.py --round 2    # obstacle boxes (adds pillar, floor, corner)
      python3 tune_rois.py --narrow     # tune the 4:3 capture instead

Controls
  1..5        pick a box (the selected one is drawn thick and white)
  arrows      move the box            (or W A S D)
  I / K       shorter / taller        (bottom edge)
  J / L       narrower / wider        (right edge)
  [ / ]       smaller / larger nudge step
  R           reset every box to the built-in default
  V           save a screenshot of the current view
  O or Enter  save to roi_config.json and keep going
  Q or Esc    quit (asks first if there are unsaved changes)

What to aim for
  left/right  On a straight, the side wall should fill most of the box and nothing else.
              A wide lens puts the walls near the edges of the frame, so these usually
              belong further out than they look.
  line        Where the orange/blue floor lines cross, close enough to the car that the
              line is clearly inside the box when a corner is reached.
  pillar      The area ahead where traffic signs appear. Wide is fine.
  floor       Directly ahead, near the car: floor lines and the magenta parking lot.
  corner      A small patch straight ahead, used only to sharpen tight corners.
  front       A narrow box at the very bottom: what the car is about to drive into.
              It should read near 0 on open track and fill with black only when a wall
              is close ahead. Compare with FRONT_BLOCK_AREA in open_challenge_R1.py.

The live numbers under each name are what the navigation code would measure right now,
normalised to 640x480 equivalents. Compare them with the thresholds in the scripts
(for example TURN_THRESH and EXIT_THRESH in open_challenge_R1.py).
"""

import argparse
import json
import sys

import cv2

from wro_functions import (CameraManager, ROI_CONFIG_PATH, area_norm, blue_mask, contours_of,
                           is_wide, load_roi_config, magenta_mask, max_contour, orange_mask,
                           red_mask, green_mask, roi_hsv_lab, roi_px, wall_mask)

DEFAULTS = {
    1: {  # open challenge
        "wide": {"left": (0.02, 0.42, 0.32, 0.58),
                 "right": (0.68, 0.42, 0.98, 0.58),
                 "line": (0.33, 0.70, 0.67, 0.86),
                 "front": (0.42, 0.86, 0.58, 0.99)},
        "narrow": {"left": (0.031, 0.354, 0.375, 0.458),
                   "right": (0.625, 0.354, 0.969, 0.458),
                   "line": (0.313, 0.625, 0.688, 0.729),
                   "front": (0.390, 0.860, 0.610, 0.990)},
    },
    2: {  # obstacle challenge
        "wide": {"left": (0.00, 0.36, 0.34, 0.58),
                 "right": (0.66, 0.36, 1.00, 0.58),
                 "pillar": (0.00, 0.20, 1.00, 0.72),
                 "floor": (0.33, 0.62, 0.67, 0.80),
                 "corner": (0.44, 0.24, 0.56, 0.30)},
        "narrow": {"left": (0.000, 0.365, 0.516, 0.552),
                   "right": (0.516, 0.365, 1.000, 0.552),
                   "pillar": (0.094, 0.250, 0.906, 0.719),
                   "floor": (0.313, 0.542, 0.688, 0.646),
                   "corner": (0.422, 0.250, 0.578, 0.292)},
    },
}

COLOURS = {"left": (0, 255, 255), "right": (0, 255, 255), "line": (255, 255, 0),
           "pillar": (255, 204, 0), "floor": (255, 0, 255), "corner": (0, 0, 255),
           "front": (0, 0, 255)}

WINDOW = "ROI tuner  -  1..5 pick   arrows move   IJKL resize   O save   Q quit"


def parse_args():
    p = argparse.ArgumentParser(description="Set the camera ROIs on the real track")
    p.add_argument("--round", type=int, choices=[1, 2], default=1, help="1 = open, 2 = obstacle")
    p.add_argument("--narrow", action="store_true", help="tune the 4:3 capture instead of 16:9")
    p.add_argument("--webcam", "-w", action="store_true", help="force the USB webcam path")
    p.add_argument("--swap-rb", dest="swap_rb", action="store_true", default=None,
                   help="force a red/blue swap")
    p.add_argument("--no-swap-rb", dest="swap_rb", action="store_false", help="never swap red/blue")
    return p.parse_args()


def measure(img, name, box, area):
    """What the navigation code would measure in this box right now."""
    hsv, lab = roi_hsv_lab(img, box)
    if name in ("left", "right"):
        return "wall", int(max_contour(contours_of(wall_mask(hsv, lab), 50), box)[0] * area)
    if name in ("line", "floor"):
        o = int(max_contour(contours_of(orange_mask(hsv, lab), 60), box)[0] * area)
        b = int(max_contour(contours_of(blue_mask(hsv, lab), 60), box)[0] * area)
        m = int(max_contour(contours_of(magenta_mask(lab), 60), box)[0] * area)
        return "O/B/M", f"{o}/{b}/{m}"
    if name == "pillar":
        r = int(max_contour(contours_of(red_mask(hsv), 60), box)[0] * area)
        g = int(max_contour(contours_of(green_mask(hsv), 60), box)[0] * area)
        return "R/G", f"{r}/{g}"
    if name in ("corner", "front"):
        return "wall", int(max_contour(contours_of(wall_mask(hsv, lab), 50), box)[0] * area)
    return "", 0


def clamp01(v):
    return max(0.0, min(1.0, v))


def main():
    args = parse_args()
    wide = not args.narrow
    key_name = "wide" if wide else "narrow"

    camera = CameraManager(force_webcam=args.webcam, wide=wide, swap_rb=args.swap_rb)
    camera.start()
    probe = None
    for _ in range(15):
        f = camera.capture_array()
        if f is not None:
            probe = f
    if probe is None:
        print("[ERROR] No frames from the camera. Try test_camera_fov.py")
        return 1

    fh, fw = probe.shape[:2]
    if is_wide(fw, fh) != wide:
        print(f"[WARNING] Asked for {'16:9' if wide else '4:3'} but the camera gave {fw}x{fh}. "
              f"Tuning that instead; saving under '{'wide' if is_wide(fw, fh) else 'narrow'}'.")
        wide = is_wide(fw, fh)
        key_name = "wide" if wide else "narrow"

    defaults = DEFAULTS[args.round][key_name]
    rois = dict(defaults)
    rois.update({k: v for k, v in load_roi_config(wide).items() if k in defaults})
    names = list(defaults)
    area = area_norm(fw, fh)

    selected = 0
    step = 0.01
    dirty = False
    print(f"[TUNER] {fw}x{fh} ({'16:9' if wide else '4:3'}), round {args.round}. "
          f"Boxes: {', '.join(names)}")
    print("[TUNER] 1..5 pick, arrows/WASD move, IJKL resize, [ ] step, R reset, O/Enter save, Q quit")

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1000, 640)
    except Exception as e:
        print(f"[ERROR] No display available ({e}). This tool needs a monitor.")
        return 1

    while True:
        img = camera.capture_array()
        if img is None:
            continue
        disp = img.copy()

        for i, name in enumerate(names):
            box = roi_px(rois[name], fw, fh)
            chosen = (i == selected)
            colour = (255, 255, 255) if chosen else COLOURS.get(name, (200, 200, 200))
            cv2.rectangle(disp, (box[0], box[1]), (box[2], box[3]), colour, 3 if chosen else 1)
            label, value = measure(img, name, box, area)
            cv2.putText(disp, f"{i + 1}:{name} {label}={value}", (box[0] + 4, max(14, box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        f = rois[names[selected]]
        cv2.putText(disp, f"[{names[selected]}] {f[0]:.3f},{f[1]:.3f},{f[2]:.3f},{f[3]:.3f}"
                          f"   step {step:.3f}{'   UNSAVED' if dirty else ''}",
                    (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(WINDOW, disp)

        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        ch = chr(k).lower() if 32 <= k < 127 else ""
        x1, y1, x2, y2 = rois[names[selected]]

        if ch in "12345" and int(ch) <= len(names):
            selected = int(ch) - 1
            continue
        if ch == "q" or k == 27:
            if dirty:
                print("[TUNER] Unsaved changes. Press O to save, or Q again to discard.")
                dirty = False
                continue
            break
        if ch == "r":
            rois = dict(defaults)
            dirty = True
            continue
        if ch == "v":
            cv2.imwrite("roi_view.jpg", disp)
            print("[TUNER] Wrote roi_view.jpg")
            continue
        if ch == "o" or k == 13:
            try:
                data = {}
                try:
                    with open(ROI_CONFIG_PATH, encoding="utf-8") as fhandle:
                        data = json.load(fhandle) or {}
                except (OSError, ValueError):
                    data = {}
                data.setdefault(key_name, {}).update({n: [round(v, 4) for v in rois[n]] for n in names})
                with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as fhandle:
                    json.dump(data, fhandle, indent=2)
                dirty = False
                print(f"[TUNER] Saved {', '.join(names)} to {ROI_CONFIG_PATH} under '{key_name}'.")
                print("[TUNER] Both challenge scripts will use these from now on.")
            except OSError as e:
                print(f"[TUNER] Could not save: {e}")
            continue
        if ch == "[":
            step = max(0.002, step / 2)
            continue
        if ch == "]":
            step = min(0.1, step * 2)
            continue

        # move: arrow keys or WASD
        dx = dy = 0.0
        if ch == "a" or k == 81:
            dx = -step
        elif ch == "d" or k == 83:
            dx = step
        elif ch == "w" or k == 82:
            dy = -step
        elif ch == "s" or k == 84:
            dy = step

        if dx or dy:
            w, h = x2 - x1, y2 - y1
            x1 = clamp01(min(x1 + dx, 1.0 - w))
            y1 = clamp01(min(y1 + dy, 1.0 - h))
            rois[names[selected]] = (x1, y1, x1 + w, y1 + h)
            dirty = True
            continue

        # resize: grows or shrinks from the bottom-right corner
        if ch == "j":
            x2 -= step
        elif ch == "l":
            x2 += step
        elif ch == "i":
            y2 -= step
        elif ch == "k":
            y2 += step
        else:
            continue
        x2, y2 = clamp01(x2), clamp01(y2)
        if x2 - x1 > 0.02 and y2 - y1 > 0.02:
            rois[names[selected]] = (x1, y1, x2, y2)
            dirty = True

    camera.stop()
    cv2.destroyAllWindows()
    print("[TUNER] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
