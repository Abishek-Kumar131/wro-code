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


class CameraManager:
    """Universal Camera abstraction supporting both Picamera2 and OpenCV USB Webcams with Zero-Lag Direct Capture."""

    def __init__(self, force_webcam=False, device_index=0):
        self.force_webcam = force_webcam
        self.device_index = device_index
        self.cap = None
        self.picam2 = None
        self.is_webcam = False

    def start(self):
        if self.force_webcam:
            self._start_webcam()
        else:
            try:
                from picamera2 import Picamera2
                print("[INFO] Initializing Picamera2 (Pi CSI Camera)...")
                self.picam2 = Picamera2()
                self.picam2.preview_configuration.main.size = (640, 480)
                self.picam2.preview_configuration.main.format = "RGB888"
                self.picam2.preview_configuration.controls.FrameRate = 30
                self.picam2.preview_configuration.align()
                self.picam2.configure("preview")
                self.picam2.start()
                self.is_webcam = False
                print("[SUCCESS] Picamera2 initialized!")
            except Exception as e:
                print(f"[INFO] Picamera2 not available ({e}). Switching to USB Webcam...")
                self._start_webcam()

    def _start_webcam(self):
        search_indices = [self.device_index, 0, 1, 2, 3, 4, 5, 6, 8]
        seen = set()
        search_indices = [x for x in search_indices if not (x in seen or seen.add(x))]

        for idx in search_indices:
            print(f"[INFO] Testing USB Webcam index {idx}...")
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                    if cap and cap.isOpened():
                        # 1. Set FOURCC to hardware MJPG FIRST (drastically reduces USB bus bandwidth from 20MB/s to 1MB/s)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        # 2. Set Frame Dimensions
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        # 3. Use double-buffering (2) to prevent Linux uvcvideo kernel driver FIFO underrun / select() timeout
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

                        # Test capture frames to verify real webcam device
                        for _ in range(3):
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                print(f"[SUCCESS] USB Webcam initialized on index {idx} (/dev/video{idx}) in MJPG 640x480 mode!")
                                self.cap = cap
                                self.device_index = idx
                                self.is_webcam = True
                                return
                        cap.release()
                except Exception:
                    pass

        print("[ERROR] Could not find any working USB webcam across indices 0-8!", file=sys.stderr)
        self.is_webcam = True
        self.cap = None

    def capture_array(self):
        if self.is_webcam:
            if self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    return frame
                else:
                    # Quick recovery retry on transient single-frame drop
                    for _ in range(2):
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            return frame
            return None
        else:
            return self.picam2.capture_array()

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
