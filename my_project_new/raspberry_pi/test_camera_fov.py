#!/usr/bin/env python3
"""
ROBOVANGUARD - WRO Future Engineers 2026
Camera field-of-view checker.

Why this exists
  The navigation code captures 640x480, which is 4:3. Most USB webcams have a 16:9
  sensor, and the common way they deliver 4:3 is to CROP the left and right edges -
  not to squeeze the wide image into a narrow one. Browser webcam test pages usually
  take the camera's default 16:9 mode, so they look much wider than our window.
  Cropping like that typically costs about 25% of the horizontal view, which is
  exactly the width you need to see both track walls at once.

  This tool captures the same scene in several modes and saves each frame, so you can
  see whether your camera crops (4:3 shows less of the room) or squeezes (4:3 shows
  the same scene, just distorted).

Usage
  python3 test_camera_fov.py                  probe the usual modes, save one frame each
  python3 test_camera_fov.py --index 2        use /dev/video2
  python3 test_camera_fov.py --live 1280x720  live window at one mode (q to quit)
  python3 test_camera_fov.py --outdir fov     where to save the frames (default: fov_test)

How to read the result
  1. Point the camera at a fixed, wide scene (a wall with objects near both edges).
     Do not move the camera between modes.
  2. Open the saved images side by side.
     - 4:3 shows LESS of the scene than 16:9  -> your camera crops. Capturing 16:9
       regains the lost width, but every ROI in the navigation code must be re-derived
       for the new frame size.
     - 4:3 shows the SAME scene, just stretched -> no FOV is lost; keep 640x480.
  3. "requested vs actual" below also tells you when the camera silently ignored the
     size you asked for and gave you something else.
"""

import argparse
import os
import subprocess
import sys

import cv2

PROBE_MODES = [
    (320, 240, "4:3"),
    (640, 480, "4:3  <- what the navigation code uses"),
    (800, 600, "4:3"),
    (1024, 768, "4:3"),
    (640, 360, "16:9"),
    (848, 480, "16:9"),
    (1280, 720, "16:9"),
    (1920, 1080, "16:9"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Compare the camera's field of view across capture modes")
    p.add_argument("--index", type=int, default=None, help="video device index (default: first that works)")
    p.add_argument("--live", help="live preview at one mode, e.g. 1280x720")
    p.add_argument("--outdir", default="fov_test", help="directory for the saved frames")
    p.add_argument("--mjpg", action="store_true", default=True, help="request MJPG (default)")
    p.add_argument("--no-mjpg", dest="mjpg", action="store_false", help="request the camera's raw format instead")
    return p.parse_args()


def open_cap(index, width, height, mjpg):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY)
    if not cap or not cap.isOpened():
        return None
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def find_camera(preferred):
    indices = [preferred] if preferred is not None else list(range(9))
    for idx in indices:
        cap = open_cap(idx, 640, 480, True)
        if cap is None:
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.size:
            return idx
    return None


def list_v4l2_modes(index):
    """Prints the camera's own list of supported modes, if v4l2-ctl is installed."""
    try:
        out = subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}", "--list-formats-ext"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        print("  (v4l2-ctl not available - skipping. Install with: sudo apt install v4l-utils)")
        return
    if out.returncode == 0 and out.stdout.strip():
        for line in out.stdout.splitlines():
            line = line.rstrip()
            if line:
                print("  " + line)
    else:
        print("  (v4l2-ctl returned nothing useful)")


def live(index, spec, mjpg):
    try:
        w, h = (int(v) for v in spec.lower().split("x"))
    except ValueError:
        print(f"[ERROR] --live wants something like 1280x720, not '{spec}'")
        return 1
    cap = open_cap(index, w, h, mjpg)
    if cap is None:
        print(f"[ERROR] Could not open /dev/video{index}")
        return 1
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[LIVE] requested {w}x{h}, actually got {aw}x{ah}. Press q to quit.")
    win = f"FOV {aw}x{ah}"
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] Frame grab failed")
            break
        cv2.putText(frame, f"{aw}x{ah}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main():
    args = parse_args()

    print("=" * 74)
    print("   ROBOVANGUARD - Camera field-of-view check")
    print("=" * 74)

    index = find_camera(args.index)
    if index is None:
        print("[ERROR] No working camera found. Check the USB cable, or pass --index N")
        return 1
    print(f"[CAMERA] Using /dev/video{index}\n")

    if args.live:
        return live(index, args.live, args.mjpg)

    print("Modes the camera reports it supports:")
    list_v4l2_modes(index)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nCapturing one frame per mode into {args.outdir}/ "
          f"({'MJPG' if args.mjpg else 'raw format'})")
    print(f"\n  {'requested':>12s}  {'actual':>11s}  {'aspect':>6s}  {'format':>6s}  file")
    print("  " + "-" * 70)

    results = []
    for w, h, note in PROBE_MODES:
        cap = open_cap(index, w, h, args.mjpg)
        if cap is None:
            print(f"  {f'{w}x{h}':>12s}  {'open failed':>11s}")
            continue
        frame = None
        for _ in range(6):          # let exposure and the mode settle
            ok, f = cap.read()
            if ok and f is not None and f.size:
                frame = f
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            pixfmt = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"
            pixfmt = "".join(c for c in pixfmt if c.isprintable())
        except Exception:
            pixfmt = "?"
        cap.release()

        if frame is None:
            print(f"  {f'{w}x{h}':>12s}  {'no frame':>11s}")
            continue

        ah_real, aw_real = frame.shape[:2]
        path = os.path.join(args.outdir, f"fov_{aw_real}x{ah_real}.jpg")
        cv2.imwrite(path, frame)
        ratio = aw_real / ah_real
        label = "16:9" if abs(ratio - 16 / 9) < 0.05 else "4:3" if abs(ratio - 4 / 3) < 0.05 else f"{ratio:.2f}"
        flag = "" if (aw_real, ah_real) == (w, h) else "   <- camera gave a different size"
        if pixfmt.upper().startswith("RGB"):
            flag += "   <- raw RGB: colours look blue unless swapped"
        print(f"  {f'{w}x{h}':>12s}  {f'{aw_real}x{ah_real}':>11s}  {label:>6s}  {pixfmt:>6s}  "
              f"{os.path.basename(path)}{flag}")
        results.append((aw_real, ah_real, label, note))

    widest_43 = max((r for r in results if r[2] == "4:3"), key=lambda r: r[0], default=None)
    widest_169 = max((r for r in results if r[2] == "16:9"), key=lambda r: r[0], default=None)

    print("\n" + "=" * 74)
    print("   WHAT TO DO NEXT")
    print("=" * 74)
    if widest_43 and widest_169:
        print(f"  Open {args.outdir}/fov_{widest_43[0]}x{widest_43[1]}.jpg (4:3) and")
        print(f"       {args.outdir}/fov_{widest_169[0]}x{widest_169[1]}.jpg (16:9) side by side.")
        print()
        print("  If the 4:3 image shows LESS of the room at the edges, your camera crops to")
        print("  make 4:3, and the navigation code is throwing away that view. Switching the")
        print("  capture to 16:9 gets it back - but every ROI box has to be re-derived for the")
        print("  new frame size, so do it BEFORE tuning the ROIs, not after.")
        print()
        print("  If both images show the same scene (the 4:3 one just looks stretched), then")
        print("  nothing is being lost and 640x480 is fine.")
    else:
        print("  Not enough modes worked to compare. Try --no-mjpg, or check the v4l2-ctl list")
        print("  above for the sizes this camera actually supports.")
    print()
    print("  Tip: keep the camera perfectly still while capturing, or the comparison lies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
