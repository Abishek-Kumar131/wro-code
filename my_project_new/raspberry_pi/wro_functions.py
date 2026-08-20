"""
ROBOVANGUARD - WRO Future Engineers 2026
Raspberry Pi 5 OpenCV Vision Functions
"""

import cv2
import numpy as np
from masks import rMagenta, rBlack


def display_roi(img, ROIs, color=(255, 204, 0)):
    """Draws Region of Interest bounding boxes on image for visual debugging."""
    for ROI in ROIs:
        img = cv2.line(img, (ROI[0], ROI[1]), (ROI[2], ROI[1]), color, 4)
        img = cv2.line(img, (ROI[0], ROI[1]), (ROI[0], ROI[3]), color, 4)
        img = cv2.line(img, (ROI[2], ROI[3]), (ROI[2], ROI[1]), color, 4)
        img = cv2.line(img, (ROI[2], ROI[3]), (ROI[0], ROI[3]), color, 4)
    return img


def find_contours(img_lab, lab_range, ROI):
    """Segment an ROI in CIELAB color space and return external contours."""
    # Segment image to ROI [x1, y1, x2, y2]
    img_segmented = img_lab[ROI[1]:ROI[3], ROI[0]:ROI[2]]

    lower_mask = np.array(lab_range[0], dtype=np.uint8)
    upper_mask = np.array(lab_range[1], dtype=np.uint8)

    # Threshold image in LAB range
    mask = cv2.inRange(img_segmented, lower_mask, upper_mask)

    kernel = np.ones((5, 5), np.uint8)

    # Erosion and dilation to filter noise
    eMask = cv2.erode(mask, kernel, iterations=1)
    dMask = cv2.dilate(eMask, kernel, iterations=1)

    # Find external contours
    contours = cv2.findContours(dMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return contours


def max_contour(contours, ROI):
    """Returns [maxArea, maxX, maxY, maxContour] for the largest contour > 100 area in ROI."""
    maxArea = 0
    maxY = 0
    maxX = 0
    mCnt = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(approx)

            # Map ROI coordinates back to full image coordinates
            x += ROI[0] + w // 2
            y += ROI[1] + h

            if area > maxArea:
                maxArea = area
                maxY = y
                maxX = x
                mCnt = cnt

    return [maxArea, maxX, maxY, mCnt]


def pOverlap(img_lab, ROI, add=False):
    """Handles black and magenta mask overlap for obstacle and parking lot detection."""
    lower_black = np.array(rBlack[0], dtype=np.uint8)
    upper_black = np.array(rBlack[1], dtype=np.uint8)
    mask_black = cv2.inRange(img_lab[ROI[1]:ROI[3], ROI[0]:ROI[2]], lower_black, upper_black)

    lower_mag = np.array(rMagenta[0], dtype=np.uint8)
    upper_mag = np.array(rMagenta[1], dtype=np.uint8)
    mask_mag = cv2.inRange(img_lab[ROI[1]:ROI[3], ROI[0]:ROI[2]], lower_mag, upper_mag)

    if not add:
        mask = cv2.subtract(mask_black, cv2.bitwise_and(mask_black, mask_mag))
    else:
        mask = cv2.add(mask_black, mask_mag)

    kernel = np.ones((5, 5), np.uint8)
    eMask = cv2.erode(mask, kernel, iterations=1)
    contours = cv2.findContours(eMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return contours


def display_variables(variables):
    """Prints debug telemetry variables on terminal using carriage returns."""
    names = list(variables.keys())
    for name in names:
        value = variables[name]
        print(f"{name}: {value}", end="\r\n")
    print("\033[F" * len(names), end="")
