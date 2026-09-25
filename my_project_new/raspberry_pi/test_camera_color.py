#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Camera colour-order check and fix.

Why this exists
  Some cameras hand their frames over with red and blue the other way round. OpenCV
  passes those through untouched, so the picture looks blue and - far worse - every
  colour threshold is wrong: orange floor lines register as blue, red pillars as green's
  opposite, and the car locks onto the wrong driving direction.

  Guessing from the camera's reported pixel format does not always work, so this tool
  measures it from a real frame instead, and saves the answer. After that the run
  scripts pick it up automatically with no command-line flag.

Usage
  1. Hold something clearly RED in front of the camera, filling the middle of the view.
     A red traffic sign from the field is ideal. Then:

         python3 test_camera_color.py

     It measures the frame, tells you which way round the camera is, and writes two
     images so you can confirm by eye.

  2. If it got it right, save the setting so every run uses it:

         python3 test_camera_color.py --save

  Other options
    --expect blue     hold a BLUE object instead of a red one
    --narrow          test the 4:3 capture instead of the wide one
    --seconds 5       how long to average over (default 3)
    --show            display a live window instead of saving images
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from wro_functions import CameraManager, CAMERA_CONFIG_PATH


def parse_args():
    p = argparse.ArgumentParser(description="Check and fix the camera's red/blue order")
    p.add_argument("--expect", choices=["red", "blue"], default="red",
                   help="colour of the object you are holding up (default: red)")
    p.add_argument("--narrow", action="store_true", help="test the 4:3 capture instead of 16:9")
    p.add_argument("--seconds", type=float, default=3.0, help="how long to average over")
    p.add_argument("--save", action="store_true", help="save the result for every future run")
    p.add_argument("--show", action="store_true", help="show a live window instead of saving images")
    p.add_argument("--outdir", default="colour_test", help="where to write the sample images")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 72)
    print("   ROBOVANGUARD - Camera colour-order check")
    print("=" * 72)
    print(f"   Hold something clearly {args.expect.upper()} in the middle of the view.")
    print(f"   Measuring for {args.seconds:.0f} s...")
    print()

    # swap_rb=False: we want the frames exactly as the camera hands them over
    camera = CameraManager(force_webcam=False, wide=not args.narrow, swap_rb=False)
    camera.start()

    end = time.time() + args.seconds
    sums = np.zeros(3, dtype=np.float64)
    frames = 0
    last = None
    while time.time() < end:
        frame = camera.capture_array()
        if frame is None:
            continue
        last = frame
        h, w = frame.shape[:2]
        patch = frame[h // 3: 2 * h // 3, w // 3: 2 * w // 3]   # middle ninth
        sums += patch.reshape(-1, 3).mean(axis=0)
        frames += 1
        if args.show:
            side = np.hstack([frame, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)])
            cv2.putText(side, "as captured", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(side, "swapped", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("as captured  |  swapped   (q to finish)", side)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    camera.stop()
    if args.show:
        cv2.destroyAllWindows()

    if not frames or last is None:
        print("[ERROR] No frames captured. Run test_camera_fov.py to check the camera.")
        return 1

    mean = sums / frames
    ch0, ch1, ch2 = mean            # channel order as delivered, before any swap
    print(f"  Frames averaged: {frames}")
    print(f"  Middle of the image, channels as delivered: [0]={ch0:6.1f}  [1]={ch1:6.1f}  [2]={ch2:6.1f}")
    print()

    # In correct BGR, a red object puts its energy in channel 2; a blue object in channel 0.
    if args.expect == "red":
        correct_is_bgr = ch2 > ch0
    else:
        correct_is_bgr = ch0 > ch2
    margin = abs(ch2 - ch0)

    if margin < 12:
        print(f"  [UNCLEAR] The two channels are only {margin:.1f} apart, so the object was not")
        print(f"            {args.expect} enough, too dark, or not filling the middle of the view.")
        print("            Move it closer, add light, and run this again.")
        verdict_swap = None
    else:
        verdict_swap = not correct_is_bgr
        if verdict_swap:
            print("  RESULT: this camera delivers RGB - red and blue ARE swapped.")
            print("          They must be corrected, or orange lines read as blue.")
        else:
            print("  RESULT: this camera delivers BGR - the colours are already correct.")
            print("          No swap needed.")

    os.makedirs(args.outdir, exist_ok=True)
    as_is = os.path.join(args.outdir, "as_captured.jpg")
    swapped = os.path.join(args.outdir, "swapped.jpg")
    cv2.imwrite(as_is, last)
    cv2.imwrite(swapped, cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    print()
    print(f"  Wrote {as_is} and {swapped} - open both; the correct one shows your")
    print(f"  {args.expect} object as {args.expect}.")

    if verdict_swap is None:
        return 1

    print()
    if args.save:
        try:
            with open(CAMERA_CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump({"swap_rb": bool(verdict_swap)}, fh, indent=2)
            print(f"  SAVED to {CAMERA_CONFIG_PATH}")
            print("  Every run now uses this automatically - no command-line flag needed.")
        except OSError as e:
            print(f"  [ERROR] Could not save: {e}")
            return 1
    else:
        print("  Nothing saved yet. If the result above matches what you see in the images:")
        print("      python3 test_camera_color.py --save")
        print(f"  Or pass {'--swap-rb' if verdict_swap else '--no-swap-rb'} to the run scripts by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
