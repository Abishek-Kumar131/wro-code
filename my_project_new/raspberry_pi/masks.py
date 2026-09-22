"""
ROBOVANGUARD - WRO Future Engineers 2026
Colour thresholds - the single place to tune colour detection.

Every range here is used by wro_functions.py. Format: (lower, upper) as [c1, c2, c3].
  HSV: H 0-180, S 0-255, V 0-255 (OpenCV)
  LAB: L, A, B each 0-255 (OpenCV 8-bit, A/B neutral = 128)
"""

# ---------------------------------------------------------------- black walls
# A pixel is wall if it matches EITHER range (covers warm and cold lighting),
# then any saturated line/pillar colour below is subtracted.
WALL_BLACK_HSV = ([0, 0, 0], [180, 140, 85])
WALL_BLACK_LAB = ([0, 95, 95], [85, 160, 165])

# Colours removed from the wall mask so floor lines and pillars are never counted as wall
WALL_EXCLUDE_HSV = [
    ([85, 60, 40], [135, 255, 255]),    # blue line
    ([8, 90, 80], [30, 255, 255]),      # orange line
    ([0, 130, 70], [8, 255, 255]),      # red (low hue)
    ([172, 130, 70], [180, 255, 255]),  # red (high hue)
    ([35, 80, 50], [85, 255, 255]),     # green
]

# ---------------------------------------------------------------- floor lines
# A pixel must match BOTH the HSV and the LAB range (rejects warm-lit white floor).
LINE_ORANGE_HSV = ([8, 120, 90], [25, 255, 255])
LINE_ORANGE_LAB = ([15, 148, 148], [235, 230, 255])

LINE_BLUE_HSV = ([100, 80, 50], [135, 255, 255])
LINE_BLUE_LAB = ([20, 110, 0], [255, 170, 115])

# ---------------------------------------------------------------- traffic signs
PILLAR_RED_HSV = [
    ([0, 120, 80], [8, 255, 255]),
    ([172, 120, 80], [180, 255, 255]),
]
PILLAR_RED_EXCLUDE_ORANGE_HSV = ([9, 80, 80], [30, 255, 255])
PILLAR_RED_MIN_ASPECT = 0.65    # bounding-box height / width; pillars are tall blocks

PILLAR_GREEN_HSV = ([35, 60, 50], [85, 255, 255])
PILLAR_GREEN_MIN_ASPECT = 0.60

# ---------------------------------------------------------------- parking lot (magenta)
rMagenta = [[0, 171, 106], [255, 195, 135]]   # LAB
lotType = "light"

# ---------------------------------------------------------------- legacy LAB ranges
# Used only by older scripts through find_contours(). Not used by open_challenge_R1.py
# or obstacle_challenge_R2.py.
rBlack = [[0, 100, 115], [65, 155, 145]]
rOrange = [[15, 148, 148], [235, 230, 255]]
rBlue = [[20, 110, 0], [255, 170, 110]]
rRed = [[0, 153, 140], [131, 198, 171]]
rGreen = [[0, 45, 0], [255, 117, 153]]
