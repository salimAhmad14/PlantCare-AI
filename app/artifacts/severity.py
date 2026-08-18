"""Green-channel lesion mask and severity grading — generated from notebook 01 v3."""
import cv2
import numpy as np

WORK_SIZE = 512
BORDER_FRACTION = 0.05
CHROMA_FLOOR = 6.0
BG_UNIFORM_MAD = 12.0
DARK_DROP = 40.0
DARK_ENCLOSURE = 0.55
DARK_MIN_AREA = 40
LEAF_CLOSE_RATIO = 0.02
LESION_SMOOTH_RATIO = 0.008
HUE_LOW = 25
HUE_HIGH = 95
SAT_MIN = 45
MIN_LESION_RATIO = 0.0008
MAX_LESION_RATIO = 0.6
MAX_BOXES = 20
BOX_PADDING_RATIO = 0.01
HEALTHY_SEVERITY = 5.0
MILD_MAX = 15.0
MODERATE_MAX = 35.0
MIN_LEAF_FRACTION = 0.04
MAX_LEAF_FRACTION = 0.96
SEVERITY_ORDER = ['Unavailable', 'Healthy', 'Mild', 'Moderate', 'High']


def _odd(value, minimum=3):
    value = max(minimum, int(value))
    return value + 1 if value % 2 == 0 else value

def fill_from_edge(mask):
    """Flood the background inward from all four corners; return the solid silhouette."""
    h, w = mask.shape
    flood = mask.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if flood[seed[1], seed[0]] == 0:
            cv2.floodFill(flood, pad, seed, 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))

def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)

def green_mask(bgr):
    """Excess-green + Otsu. Kept only as the fallback for non-uniform backgrounds."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min(bgr.shape[:2]) * 0.012),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return largest_component(mask)

def background_stats(lab, border=BORDER_FRACTION):
    """Median and MAD of the border strip — the model of 'what background looks like'."""
    h, w = lab.shape[:2]
    b = max(2, int(min(h, w) * border))
    frame = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    median = np.median(frame, axis=0)
    mad = np.median(np.abs(frame - median), axis=0)
    return median, mad

def rescue_dark_tissue(lab, core, bg_median, drop=DARK_DROP,
                       enclosure=DARK_ENCLOSURE, min_area=DARK_MIN_AREA):
    """Near-black necrosis has no chroma, so the background rule cannot see it.

    Keep a dark blob if most of its border touches leaf tissue (necrosis sits IN the leaf)
    and drop it if it opens onto the background (that is a cast shadow)."""
    lightness = lab[:, :, 0]
    dark = ((lightness < bg_median[0] - drop) & (core == 0)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    keep = np.zeros_like(dark)
    h, w = dark.shape

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue                       # touches the frame -> it is the scene, not a lesion
        blob = (labels == i).astype(np.uint8) * 255
        ring = cv2.subtract(cv2.dilate(blob, ring_kernel), blob)
        ring_pixels = np.count_nonzero(ring)
        if ring_pixels and np.count_nonzero(cv2.bitwise_and(ring, core)) / ring_pixels >= enclosure:
            keep = cv2.bitwise_or(keep, blob)
    return keep

def leaf_mask(bgr):
    """The whole leaf — green, brown, yellow and black tissue alike."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_median, bg_mad = background_stats(lab)
    chroma_mad = float(bg_mad[1] + bg_mad[2])
    uniform = chroma_mad <= BG_UNIFORM_MAD

    green = green_mask(bgr)

    if uniform:
        # chroma distance ONLY — including lightness would swallow the cast shadow
        distance = np.sqrt(((lab[:, :, 1:] - bg_median[1:]) ** 2).sum(axis=2))
        u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cut, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        chroma = (((u8 >= cut) & (distance >= CHROMA_FLOOR)).astype(np.uint8)) * 255
        core = largest_component(cv2.bitwise_or(chroma, green))
        core = cv2.bitwise_or(core, rescue_dark_tissue(lab, core, bg_median))
    else:
        core = largest_component(green)

    h, w = core.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min(h, w) * LEAF_CLOSE_RATIO),) * 2)
    solid = fill_from_edge(cv2.morphologyEx(fill_from_edge(core), cv2.MORPH_CLOSE, k))
    return solid, green, {"uniform_background": bool(uniform),
                          "background_chroma_mad": round(chroma_mad, 2)}

def healthy_tissue_mask(bgr, leaf):
    """Green, saturated tissue inside the leaf. Everything else inside it is a lesion."""
    hue, sat, _ = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    b, g, r = cv2.split(bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    healthy = ((hue >= HUE_LOW) & (hue <= HUE_HIGH) & (sat >= SAT_MIN) & (exg > 0))
    return cv2.bitwise_and(healthy.astype(np.uint8) * 255, leaf)

def analyse_leaf(bgr):
    """Leaf mask, healthy mask, lesion mask, boxes (original scale) and severity %."""
    h0, w0 = bgr.shape[:2]
    scale = WORK_SIZE / max(h0, w0)
    work = (cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1 else bgr.copy())
    h, w = work.shape[:2]

    leaf, green, info = leaf_mask(work)
    healthy = healthy_tissue_mask(work, leaf)
    lesion = cv2.bitwise_and(leaf, cv2.bitwise_not(healthy))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(min(h, w) * LESION_SMOOTH_RATIO),) * 2)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k, iterations=1)
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, k, iterations=2)

    leaf_area = max(1, int(np.count_nonzero(leaf)))
    severity = 100.0 * np.count_nonzero(lesion) / leaf_area

    n, labels, stats, _ = cv2.connectedComponentsWithStats(lesion, 8)
    pad = max(2, int(min(h, w) * BOX_PADDING_RATIO))
    scored = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        ratio = area / leaf_area
        if ratio < MIN_LESION_RATIO or ratio > MAX_LESION_RATIO:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if bw < 4 or bh < 4:
            continue
        scored.append((area, (max(0, x - pad), max(0, y - pad),
                              min(w - 1, x + bw + pad), min(h - 1, y + bh + pad))))

    scored.sort(key=lambda t: t[0], reverse=True)
    inv = 1.0 / scale if scale < 1 else 1.0
    boxes = [tuple(int(v * inv) for v in box) for _, box in scored[:MAX_BOXES]]

    result = {
        "leaf_mask": leaf,
        "green_mask": green,
        "healthy_mask": healthy,
        "lesion_mask": lesion,
        "boxes": boxes,
        "severity": severity,
        "work_shape": (h, w),
        # share of the frame the leaf covers. Section 11's first screen reads this to
        # decide whether there is a leaf in the picture at all.
        "leaf_fraction": float(leaf_area) / float(h * w),
    }
    result.update(info)
    return result

def severity_level(percent,
                   healthy_max=HEALTHY_SEVERITY,
                   mild_max=MILD_MAX,
                   moderate_max=MODERATE_MAX):
    """% of leaf area covered by lesions  ->  severity band."""
    if percent is None or not np.isfinite(percent):
        return "Unavailable"
    if percent < healthy_max:
        return "Healthy"
    if percent < mild_max:
        return "Mild"
    if percent < moderate_max:
        return "Moderate"
    return "High"

def leaf_mask_quality(analysis):
    """Fraction of the frame the leaf mask claims. Outside the sane range = mask failure."""
    if "leaf_fraction" in analysis:            # one source of truth, no drift
        fraction = float(analysis["leaf_fraction"])
    else:
        h, w = analysis["work_shape"]
        fraction = np.count_nonzero(analysis["leaf_mask"]) / float(max(1, h * w))
    return float(fraction), bool(MIN_LEAF_FRACTION <= fraction <= MAX_LEAF_FRACTION)

def assess_severity(bgr, analysis=None):
    """Full severity read-out for one leaf image."""
    if analysis is None:
        analysis = analyse_leaf(bgr)

    leaf_fraction, mask_ok = leaf_mask_quality(analysis)
    leaf_area = max(1, int(np.count_nonzero(analysis["leaf_mask"])))

    count, _, stats, _ = cv2.connectedComponentsWithStats(analysis["lesion_mask"], 8)
    blob_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, count)]
    keep = [a for a in blob_areas if MIN_LESION_RATIO <= a / leaf_area <= MAX_LESION_RATIO]
    largest_pct = float(100.0 * max(keep) / leaf_area) if keep else 0.0

    percent = float(analysis["severity"]) if mask_ok else None
    level = severity_level(percent) if mask_ok else "Unavailable"

    return {
        "severity_percent": None if percent is None else round(percent, 2),
        "severity_level": level,
        "lesion_count": len(keep),
        "largest_lesion_pct": round(largest_pct, 2),
        "leaf_fraction": round(leaf_fraction, 3),
        "mask_ok": bool(mask_ok),
    }
