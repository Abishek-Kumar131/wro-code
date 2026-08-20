"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 OpenCV Vision Functions & Drawing Helpers
(Matching helpers & drawing routines from my_old_contour_colorvals_crt.py)
"""

import cv2
import numpy as np
from masks import rBlack, rMagenta


def morphology_clean(mask, ksize=5, iterations=1):
    """Applies morphological close operation to filter noise."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def find_contours(img_lab, lab_range, ROI, min_area=60):
    """Segment an ROI in CIELAB color space, apply Gaussian blur & MORPH_CLOSE, returning filtered contours."""
    x1, y1, x2, y2 = ROI
    img_segmented = img_lab[y1:y2, x1:x2]

    lower_mask = np.array(lab_range[0], dtype=np.uint8)
    upper_mask = np.array(lab_range[1], dtype=np.uint8)

    mask = cv2.inRange(img_segmented, lower_mask, upper_mask)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = morphology_clean(mask, 5, 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > min_area]


def max_contour(contours, ROI=[0, 0, 0, 0]):
    """Returns [maxArea, maxX, maxY, maxContour] for the largest contour in ROI."""
    if not contours:
        return [0, 0, 0, None]

    maxArea = 0
    maxY = 0
    maxX = 0
    mCnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > maxArea:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            x += ROI[0] + w // 2
            y += ROI[1] + h
            maxArea = int(area)
            maxY = y
            maxX = x
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
