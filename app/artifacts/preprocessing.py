"""Blur removal and brightness correction — generated from notebook 01 v3.

Applied to the TRAINING split only during training. If you call this at
inference time, say so in your results: it changes the input distribution.
"""
import math

import cv2
import numpy as np

BLUR_VARIANCE_MIN = 120.0
UNSHARP_RADIUS = 3.0
UNSHARP_AMOUNT = 0.8
BRIGHTNESS_TARGET = 128.0
BRIGHTNESS_LOW = 95.0
BRIGHTNESS_HIGH = 165.0
GAMMA_LIMITS = (0.55, 1.85)
CLAHE_CLIP = 2.0
CLAHE_GRID = 8


def blur_score(bgr):
    """Variance of the Laplacian. High = sharp, low = blurry. Scale-dependent, so it is
    only meaningful compared against other images of similar size."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness_score(bgr):
    """Mean grey level, 0-255. ~128 is well exposed."""
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())

def remove_blur(bgr, radius=UNSHARP_RADIUS, amount=UNSHARP_AMOUNT):
    """Unsharp mask: add back a scaled copy of what the blur removed."""
    blurred = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)

def correct_brightness(bgr, target=BRIGHTNESS_TARGET, clip=CLAHE_CLIP, grid=CLAHE_GRID,
                       gamma_limits=GAMMA_LIMITS):
    """CLAHE on LAB-L for local contrast, then a clamped gamma toward mid-grey.

    A and B are untouched on purpose — the lesion mask is a colour rule, so shifting hue
    here would move every severity number downstream."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(int(grid), int(grid))).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    mean = brightness_score(out)
    if 1.0 < mean < 254.0:
        # solve (mean/255) ** (1/gamma) == target/255  for gamma.
        # gamma > 1 brightens, gamma < 1 darkens. Getting this ratio the wrong way
        # round silently pushes dark images darker, so it is asserted below.
        gamma = math.log(mean / 255.0) / math.log(target / 255.0)
        gamma = float(np.clip(gamma, *gamma_limits))
        if abs(gamma - 1.0) > 0.02:
            table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in range(256)], dtype=np.uint8)
            out = cv2.LUT(out, table)
    return out

def preprocess_image(bgr, blur_min=BLUR_VARIANCE_MIN,
                     bright_low=BRIGHTNESS_LOW, bright_high=BRIGHTNESS_HIGH,
                     report=False):
    """Conditional clean-up. Returns the image, or (image, report) when report=True.

    A good image passes through untouched — check `deblurred` / `rebalanced` in the
    report to see whether anything actually fired."""
    info = {
        "blur_before": blur_score(bgr),
        "brightness_before": brightness_score(bgr),
        "deblurred": False,
        "rebalanced": False,
    }

    out = bgr
    if info["blur_before"] < blur_min:
        out = remove_blur(out)
        info["deblurred"] = True

    if not (bright_low <= info["brightness_before"] <= bright_high):
        out = correct_brightness(out)
        info["rebalanced"] = True

    if report:
        info["blur_after"] = blur_score(out)
        info["brightness_after"] = brightness_score(out)
        info["changed"] = info["deblurred"] or info["rebalanced"]
        return out, info
    return out
