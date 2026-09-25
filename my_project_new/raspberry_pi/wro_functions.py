import json
import os
import sys
import time

# Suppress OpenCV C++ V4L2 backend warning spam
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np

import masks as M
from masks import rMagenta

KERNEL5 = np.ones((5, 5), np.uint8)


# Written by test_camera_color.py, read on every run, so a camera that delivers RGB is
# corrected automatically without passing a flag.
CAMERA_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")


def load_camera_config():
    try:
        with open(CAMERA_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# Written by tune_rois.py. Holds ROI fractions measured on the real track, per aspect:
#   {"wide": {"left": [x1, y1, x2, y2], ...}, "narrow": {...}}
ROI_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roi_config.json")


def load_roi_config(wide):
    """ROI overrides measured on the track for this aspect, or {} if none saved."""
    try:
        with open(ROI_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    section = data.get("wide" if wide else "narrow", {})
    out = {}
    if isinstance(section, dict):
        for name, box in section.items():
            if isinstance(box, (list, tuple)) and len(box) == 4:
                out[name] = tuple(float(v) for v in box)
    return out


def apply_roi_config(defaults, wide, announce=True):
    """Merges saved ROI fractions over the built-in defaults."""
    rois = dict(defaults)
    saved = load_roi_config(wide)
    if saved:
        rois.update(saved)
        if announce:
            print(f"[ROI] Using measured ROIs from roi_config.json for "
                  f"{'16:9' if wide else '4:3'}: {', '.join(sorted(saved))}")
    return rois


def is_wide(w, h):
    """True if this frame is 16:9 (the camera's full width) rather than a 4:3 crop."""
    return h > 0 and abs((w / float(h)) - (16.0 / 9.0)) < 0.12


class CameraManager:
    """
    Camera abstraction for Picamera2 and USB webcams.

    Captures 16:9 by default, because asking a 16:9 webcam for a 4:3 size (like the old
    fixed 640x480) makes most cameras CROP the left and right edges off the sensor -
    throwing away about 25% of the horizontal field of view, which is exactly the width
    needed to see both track walls at once.

    Frames larger than max_width are downscaled, so a camera that only offers 1280x720
    still costs the same CPU as a small mode. Use wide=False to go back to 4:3.
    """

    MODES_WIDE = [(848, 480), (960, 540), (1280, 720), (640, 360)]
    MODES_43 = [(640, 480), (800, 600), (320, 240)]

    def __init__(self, force_webcam=False, device_index=0, wide=True, max_width=960, swap_rb=None):
        self.force_webcam = force_webcam
        self.device_index = device_index
        self.wide = wide
        self.max_width = max_width
        self.swap_rb = swap_rb      # None = decide from the camera's pixel format
        self.cap = None
        self.picam2 = None
        self.is_webcam = False
        self.width = 0
        self.height = 0
        self.fourcc = "?"
        self._resize_to = None      # (w, h) when the captured frame must be downscaled
        self._swap_rb = bool(swap_rb)

    def start(self):
        if self.force_webcam:
            self._start_webcam()
        else:
            try:
                from picamera2 import Picamera2
                size = (1280, 720) if self.wide else (640, 480)

                # Only use Picamera2 for a real CSI camera.
                # libcamera also lists USB cameras through its uvcvideo pipeline, and
                # driving one that way delivers RGB where a CSI sensor delivers BGR - so
                # every colour comes out with red and blue swapped. It can also hold
                # /dev/videoN open and lock OpenCV out.
                # A USB camera's Id is its USB path, e.g.
                #   /base/axi/pcie@1000120000/rp1/usb@300000-1:1.0-1bd0:2282
                # while a CSI sensor sits on i2c, e.g. .../i2c@88000/imx708@1a
                infos = Picamera2.global_camera_info()

                def looks_like_csi(cam):
                    ident = f"{cam.get('Id', '')} {cam.get('Model', '')}".lower()
                    return not any(tag in ident for tag in ("usb", "uvc", "video4linux"))

                csi = [i for i, cam in enumerate(infos) if looks_like_csi(cam)]
                if infos:
                    print("[INFO] libcamera sees: "
                          + ", ".join(f"{c.get('Model', '?')} ({'CSI' if looks_like_csi(c) else 'USB'})"
                                      for c in infos))
                if not csi:
                    raise RuntimeError("no CSI camera; USB cameras are handled better by OpenCV")

                print(f"[INFO] Initializing Picamera2 (Pi CSI Camera) at {size[0]}x{size[1]}...")
                self.picam2 = Picamera2(csi[0])
                self.picam2.preview_configuration.main.size = size
                self.picam2.preview_configuration.main.format = "RGB888"
                try:
                    self.picam2.preview_configuration.controls.FrameRate = 30
                except Exception:
                    pass
                self.picam2.preview_configuration.align()
                self.picam2.configure("preview")
                self.picam2.start()
                self.is_webcam = False
                probe = self.picam2.capture_array()
                self._note_size(probe)
                print("[SUCCESS] Picamera2 initialized!")
            except Exception as e:
                print(f"[INFO] Picamera2 not available ({e}). Switching to USB Webcam...")
                if self.picam2 is not None:
                    # Must release it, or the camera stays busy and OpenCV cannot open it
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                    self.picam2 = None
                self._start_webcam()

    def _note_size(self, frame):
        """Records the frame size actually delivered, and sets up downscaling if needed."""
        if frame is None:
            return
        h, w = frame.shape[:2]
        if self.max_width and w > self.max_width:
            scale = self.max_width / float(w)
            self._resize_to = (int(w * scale), int(h * scale))
            self.width, self.height = self._resize_to
            print(f"[CAMERA] Capturing {w}x{h}, downscaling to {self.width}x{self.height}")
        else:
            self._resize_to = None
            self.width, self.height = w, h
            print(f"[CAMERA] Capturing {w}x{h}")

        # Everything downstream assumes OpenCV's usual BGR order. Most webcam formats
        # (MJPG, YUYV) are converted to BGR for us, but a camera handing over raw RGB
        # is passed straight through - which makes the whole image look blue.
        if self.swap_rb is None:
            saved = load_camera_config()
            if "swap_rb" in saved:
                self._swap_rb = bool(saved["swap_rb"])
                why = "measured by test_camera_color.py"
            else:
                self._swap_rb = self.fourcc.upper().startswith("RGB")
                why = f"guessed from pixel format {self.fourcc}"
        else:
            why = "set on the command line"

        if self._swap_rb:
            print(f"[CAMERA] Pixel format {self.fourcc}; swapping red/blue ({why})")
        elif self.is_webcam:
            print(f"[CAMERA] Pixel format {self.fourcc}; no red/blue swap ({why}). "
                  f"If the picture looks blue, run: python3 test_camera_color.py --save")

        if self.wide and not is_wide(self.width, self.height):
            print("[CAMERA WARNING] Asked for a 16:9 mode but got "
                  f"{self.width}x{self.height}. This camera is cropping the sides off the "
                  "sensor, so part of the field of view is lost. Run test_camera_fov.py "
                  "to see which modes it really supports.", file=sys.stderr)

    def _open(self, idx, backend, w, h):
        cap = cv2.VideoCapture(idx, backend)
        if not cap or not cap.isOpened():
            return None, None
        # FOURCC first: MJPG cuts USB bandwidth from ~20MB/s to ~1MB/s
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Double-buffering avoids uvcvideo FIFO underrun / select() timeouts
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        frame = None
        for _ in range(3):
            ret, f = cap.read()
            if ret and f is not None and f.size > 0:
                frame = f
        if frame is None:
            cap.release()
            return None, None
        return cap, frame

    @staticmethod
    def _fourcc_of(cap):
        """The pixel format the camera actually gave us, e.g. MJPG, YUYV, RGB3."""
        try:
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
        except Exception:
            return "?"
        if not code:
            return "?"
        text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
        return "".join(c for c in text if c.isprintable()).strip() or "?"

    def _start_webcam(self):
        try:                                   # keep OpenCV's per-attempt warnings quiet
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:
            pass

        # On Linux only probe video devices that actually exist, instead of 0-8 blindly
        if sys.platform.startswith("linux"):
            import glob as _glob
            present = sorted(int(p.rsplit("video", 1)[1]) for p in _glob.glob("/dev/video*")
                             if p.rsplit("video", 1)[-1].isdigit())
            search_indices = [self.device_index] + present if present else [self.device_index]
        else:
            search_indices = [self.device_index, 0, 1, 2, 3, 4, 5, 6, 8]
        seen = set()
        search_indices = [x for x in search_indices if not (x in seen or seen.add(x))]
        modes = self.MODES_WIDE if self.wide else self.MODES_43
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if sys.platform.startswith("linux") else [cv2.CAP_ANY]

        # Probe every mode first and release it again, then pick the best one and reopen.
        # Ranking, in order: the aspect we asked for, then MJPG.
        # MJPG matters because OpenCV decodes it to BGR just like the old 640x480 capture
        # did. Some cameras only offer raw RGB at wide sizes, and raw RGB comes through
        # with red and blue swapped, which also makes orange floor lines look blue.
        candidates = []
        for idx in search_indices:
            print(f"[INFO] Testing USB Webcam index {idx}...")
            for backend in backends:
                for order, (w, h) in enumerate(modes):
                    try:
                        cap, frame = self._open(idx, backend, w, h)
                    except Exception:
                        cap, frame = None, None
                    if cap is None:
                        continue
                    fh, fw = frame.shape[:2]
                    fourcc = self._fourcc_of(cap)
                    cap.release()
                    candidates.append({"idx": idx, "backend": backend, "w": w, "h": h,
                                       "fw": fw, "fh": fh, "fourcc": fourcc, "order": order})
                if candidates:
                    break          # this backend works on this device; no need for the other
            if candidates:
                break              # first working device wins

        if not candidates:
            print("[ERROR] Could not find any working USB webcam!", file=sys.stderr)
            self.is_webcam = True
            self.cap = None
            return

        def aspect_ok(c):
            return is_wide(c["fw"], c["fh"]) if self.wide else not is_wide(c["fw"], c["fh"])

        def rank(c):
            return (aspect_ok(c), c["fourcc"].upper() == "MJPG", -c["order"])

        for c in sorted(candidates, key=rank, reverse=True):
            cap, frame = self._open(c["idx"], c["backend"], c["w"], c["h"])
            if cap is None:
                continue
            self.cap = cap
            self.device_index = c["idx"]
            self.is_webcam = True
            self.fourcc = self._fourcc_of(cap)
            fh, fw = frame.shape[:2]
            print(f"[SUCCESS] USB Webcam on index {c['idx']} (/dev/video{c['idx']}), "
                  f"asked {c['w']}x{c['h']}, got {fw}x{fh} in {self.fourcc}")
            if self.wide and not aspect_ok(c):
                others = ", ".join(sorted({f"{x['fw']}x{x['fh']} {x['fourcc']}" for x in candidates}))
                print(f"[WARNING] No 16:9 mode worked on this camera. It offered: {others}")
            self._note_size(frame)
            return

        print("[ERROR] Every camera mode failed on reopen!", file=sys.stderr)
        self.is_webcam = True
        self.cap = None

    def capture_array(self):
        frame = None
        if self.is_webcam:
            if self.cap is not None:
                ret, f = self.cap.read()
                if ret and f is not None:
                    frame = f
                else:
                    # Quick recovery retry on transient single-frame drop
                    for _ in range(2):
                        ret, f = self.cap.read()
                        if ret and f is not None:
                            frame = f
                            break
        else:
            frame = self.picam2.capture_array()

        if frame is not None:
            if self._swap_rb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self._resize_to is not None:
                frame = cv2.resize(frame, self._resize_to, interpolation=cv2.INTER_AREA)
        return frame

    def stop(self):
        if self.is_webcam and self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        elif self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass


# ============================================================================
# Colour masks. Convert an ROI once with roi_hsv_lab(), then build any number of
# masks from the same conversion. All thresholds live in masks.py.
# ============================================================================

def roi_px(frac, w, h):
    """
    Turns a fraction-of-frame ROI (x1, y1, x2, y2, each 0..1) into a pixel box for this
    frame size. ROIs are stored as fractions so that changing the capture resolution
    does not silently move every box to the wrong part of the world.
    """
    x1 = max(0, min(w - 2, int(round(frac[0] * w))))
    x2 = max(x1 + 1, min(w, int(round(frac[2] * w))))
    y1 = max(0, min(h - 2, int(round(frac[1] * h))))
    y2 = max(y1 + 1, min(h, int(round(frac[3] * h))))
    return [x1, y1, x2, y2]


def area_norm(w, h):
    """
    Scale factor that converts a measured contour area into '640x480 equivalent pixels',
    so the area thresholds in the navigation code keep their meaning at any resolution.

    On a 16:9 frame there is a second factor. The wide ROI fractions are narrower than
    the 4:3 ones (a 4:3 crop is 75% of the 16:9 width), so the ROI stays about the same
    size in pixels while the frame itself got wider. Dividing by frame area alone would
    then make the same wall measure about 25% smaller, quietly shifting every threshold.
    """
    base = (640.0 * 480.0) / float(max(1, w * h))
    return base / 0.75 if is_wide(w, h) else base


def _in(img, rng):
    return cv2.inRange(img, np.array(rng[0], dtype=np.uint8), np.array(rng[1], dtype=np.uint8))


def roi_hsv_lab(img_bgr, roi):
    """Returns (hsv, lab) for one ROI [x1, y1, x2, y2]. Convert once, reuse for every mask."""
    x1, y1, x2, y2 = roi
    patch = img_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2HSV), cv2.cvtColor(patch, cv2.COLOR_BGR2Lab)


def wall_mask(hsv, lab):
    """Black wall pixels, with floor-line and pillar colours removed."""
    mask = cv2.bitwise_or(_in(hsv, M.WALL_BLACK_HSV), _in(lab, M.WALL_BLACK_LAB))
    for rng in M.WALL_EXCLUDE_HSV:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(_in(hsv, rng)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5)


def orange_mask(hsv, lab):
    mask = cv2.bitwise_and(_in(hsv, M.LINE_ORANGE_HSV), _in(lab, M.LINE_ORANGE_LAB))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def blue_mask(hsv, lab):
    mask = cv2.bitwise_and(_in(hsv, M.LINE_BLUE_HSV), _in(lab, M.LINE_BLUE_LAB))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def red_mask(hsv):
    mask = _in(hsv, M.PILLAR_RED_HSV[0])
    for rng in M.PILLAR_RED_HSV[1:]:
        mask = cv2.bitwise_or(mask, _in(hsv, rng))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(_in(hsv, M.PILLAR_RED_EXCLUDE_ORANGE_HSV)))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5)


def green_mask(hsv):
    return cv2.morphologyEx(_in(hsv, M.PILLAR_GREEN_HSV), cv2.MORPH_CLOSE, KERNEL5)


def magenta_mask(lab):
    mask = cv2.GaussianBlur(_in(lab, rMagenta), (7, 7), 0)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL5)


def contours_of(mask, min_area=0, min_aspect=0.0):
    """External contours of a mask, filtered by area and (optionally) height/width aspect."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) <= min_area:
            continue
        if min_aspect > 0:
            _, _, w, h = cv2.boundingRect(c)
            if h / max(1.0, float(w)) < min_aspect:
                continue
        out.append(c)
    return out


# ============================================================================
# Backwards-compatible wrappers (older scripts import these names)
# ============================================================================

def morphology_clean(mask, ksize=5, iterations=1):
    """Applies morphological close operation to filter noise."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def find_black_wall_contours(img_bgr, ROI, min_area=50):
    return contours_of(wall_mask(*roi_hsv_lab(img_bgr, ROI)), min_area)


def find_red_pillar_contours(img_bgr, ROI, min_area=70):
    hsv, _ = roi_hsv_lab(img_bgr, ROI)
    return contours_of(red_mask(hsv), min_area - 1, M.PILLAR_RED_MIN_ASPECT)


def find_green_pillar_contours(img_bgr, ROI, min_area=70):
    hsv, _ = roi_hsv_lab(img_bgr, ROI)
    return contours_of(green_mask(hsv), min_area - 1, M.PILLAR_GREEN_MIN_ASPECT)


def find_orange_line_contours(img_bgr, ROI, min_area=100):
    return contours_of(orange_mask(*roi_hsv_lab(img_bgr, ROI)), min_area)


def find_blue_line_contours(img_bgr, ROI, min_area=100):
    return contours_of(blue_mask(*roi_hsv_lab(img_bgr, ROI)), min_area)


def find_contours(img_lab, lab_range, ROI, min_area=60):
    """Segment an ROI of a full-frame LAB image with one LAB range (legacy path)."""
    x1, y1, x2, y2 = ROI
    mask = _in(img_lab[y1:y2, x1:x2], lab_range)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = morphology_clean(mask, 5, 1)
    return contours_of(mask, min_area)


def max_contour(contours, ROI=(0, 0, 0, 0)):
    """Returns [maxArea, centerX, bottomY, contour] of the largest contour, in full-frame coordinates."""
    if not contours:
        return [0, 0, 0, None]

    maxArea = 0
    maxY = 0
    maxX = 0
    mCnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            x, y, w, h = cv2.boundingRect(cnt)
            maxArea = int(area)
            maxX = x + ROI[0] + w // 2
            maxY = y + ROI[1] + h
            mCnt = cnt

    return [maxArea, maxX, maxY, mCnt]


def draw_roi(frame, roi, color=(0, 255, 255), thick=2):
    """Draws ROI boundary rectangle on frame."""
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)


def draw_offset_contours(frame, contours, roi, color, thick=2):
    """Draws contours offset to their correct full-frame coordinates within ROI."""
    if not contours:
        return
    x1, y1, _, _ = roi
    offset = np.array([[x1, y1]], dtype=np.int32)
    shifted = [cnt + offset for cnt in contours]
    cv2.drawContours(frame, shifted, -1, color, thick)


def display_variables(variables):
    """Prints debug telemetry variables on terminal using carriage returns."""
    names = list(variables.keys())
    for name in names:
        value = variables[name]
        print(f"{name}: {value}", end="\r\n")
    print("\033[F" * len(names), end="")


# ============================================================================
# Run helpers shared by open_challenge_R1.py and obstacle_challenge_R2.py
# ============================================================================

class FpsCounter:
    def __init__(self):
        self.fps = 0.0
        self._last = time.time()

    def tick(self):
        now = time.time()
        dt = now - self._last
        self._last = now
        if dt > 0:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        return self.fps


class Ultrasonics:
    """
    Reads the ESP32's ultrasonic telemetry safely.

    Two rules that the old code got wrong:
      - 0 means "nothing within range", NOT "a wall is touching us". Every test here
        requires a positive reading, so open track can never look like a collision.
      - near() only reports True after the condition holds on `confirm` consecutive
        updates, which rejects the single-frame spikes these sensors produce.

    Call update() once per camera frame, then near()/get() as often as you like.
    """

    KEYS = ("f", "f1", "f2", "l", "r", "b")

    def __init__(self, link, confirm=2, max_valid=400):
        self.link = link
        self.confirm = confirm
        self.max_valid = max_valid
        self.values = {k: 0 for k in self.KEYS}
        self._counts = {}
        self._frame = 0

    def update(self):
        data = self.link.get_us_data()
        for k in self.KEYS:
            try:
                self.values[k] = int(data.get(k, 0) or 0)
            except (TypeError, ValueError):
                self.values[k] = 0
        # Advance every counter here, not in near(), so the count tracks sensor updates
        # even on frames where the caller does not test that condition.
        for slot in self._counts:
            key, cm = slot
            self._counts[slot] = self._counts[slot] + 1 if self._hit(key, cm) else 0
        self._frame += 1
        return self.values

    def get(self, key):
        return self.values.get(key, 0)

    def valid(self, key):
        """True if this sensor returned a usable reading (not 0, not absurd)."""
        return 0 < self.values.get(key, 0) <= self.max_valid

    def _hit(self, key, cm):
        return self.valid(key) and self.values[key] <= cm

    def near(self, key, cm):
        """True once this sensor has read 0 < value <= cm on `confirm` consecutive updates."""
        slot = (key, cm)
        if slot not in self._counts:
            self._counts[slot] = 1 if self._hit(key, cm) else 0
        return self._counts[slot] >= self.confirm

    def clear_ahead(self, cm):
        """True if none of the three front sensors sees anything closer than cm."""
        return all(self.values[k] == 0 or self.values[k] >= cm for k in ("f", "f1", "f2"))

    def text(self):
        return " ".join(f"{k.upper()}:{self.values[k]}" for k in self.KEYS)


def _make_button_reader(gpio_pin, active_high):
    """Returns a function that reports whether the start button is pressed."""
    try:
        from gpiozero import Button
        button = Button(gpio_pin, pull_up=not active_high, bounce_time=0.02)
        print(f"[GPIO] Start button on GPIO {gpio_pin} (gpiozero).")
        return lambda: button.is_pressed
    except Exception:
        pass
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN if active_high else GPIO.PUD_UP)
        print(f"[GPIO] Start button on GPIO {gpio_pin} (RPi.GPIO).")
        pressed_level = GPIO.HIGH if active_high else GPIO.LOW
        return lambda: GPIO.input(gpio_pin) == pressed_level
    except Exception as e:
        print(f"[GPIO] No GPIO library available ({e}). Use ENTER or the 's' key instead.")
        return lambda: False


def wait_for_button_press(gpio_pin=17, active_high=False, camera=None, show_display=False, window_name=""):
    """
    Blocks until the start button (or ENTER in a terminal, or 's'/ENTER/SPACE in the display
    window) is pressed. Keeps grabbing camera frames while waiting so auto-exposure stays
    settled and the first frame after the start is fresh.
    """
    read_button = _make_button_reader(gpio_pin, active_high)
    interactive = sys.stdin.isatty() and not sys.platform.startswith("win")

    t0 = time.time()
    while read_button() and time.time() - t0 < 3.0:   # button held during boot: wait for release
        time.sleep(0.02)

    print("=" * 65)
    print(f"[READY] Press the start button (GPIO {gpio_pin})" + (" or ENTER" if interactive else ""))
    print("=" * 65, flush=True)

    if interactive:
        import select

    while True:
        if read_button():
            time.sleep(0.03)
            if read_button():
                return True

        if interactive and sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True

        if camera is not None:
            frame = camera.capture_array()
            if show_display and frame is not None:
                disp = frame.copy()
                cv2.putText(disp, "READY - press button or 's'", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('s'), ord('S'), 13, 32):
                    return True
                if key in (ord('q'), 27):
                    print("[ABORT] Cancelled from display window.")
                    sys.exit(0)
        else:
            time.sleep(0.02)
