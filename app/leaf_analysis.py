"""Leaf analysis bridge.

Does NOT reimplement the mask maths. It loads `artifacts/severity.py` - the exact
module exported from the training notebook - so the web app and the notebook can
never drift apart. If that file is missing the app refuses to start rather than
silently falling back to different numbers.
"""

import base64
import importlib.util
import sys

import cv2
import numpy as np

import config

_sev = None


def _load_severity():
    global _sev
    if _sev is not None:
        return _sev
    if not config.SEVERITY_MODULE.exists():
        raise FileNotFoundError(
            f"severity.py not found at {config.SEVERITY_MODULE}\n"
            "Copy it out of PlantCare_CNN/ on Kaggle into the artifacts/ folder.")
    spec = importlib.util.spec_from_file_location("severity", config.SEVERITY_MODULE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["severity"] = mod
    spec.loader.exec_module(mod)
    _sev = mod
    return _sev


def to_data_uri(bgr, quality=85, max_side=900):
    """Encode a BGR array as a data: URI so templates need no static files."""
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _mask_to_bgr(mask, shape):
    m = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def overlay_lesions(bgr, lesion_mask, boxes=None, alpha=0.45, max_boxes=12):
    """Red translucent lesion overlay + yellow region boxes."""
    out = bgr.copy()
    if lesion_mask is not None and np.count_nonzero(lesion_mask):
        full = cv2.resize(lesion_mask, (bgr.shape[1], bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        red = np.zeros_like(out)
        red[:, :, 2] = 255
        out = np.where(full[:, :, None] > 0,
                       cv2.addWeighted(out, alpha, red, 1 - alpha, 0), out)
    for box in (boxes or [])[:max_boxes]:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    return out


def analyse(bgr, crop=None):
    """Run severity + build the side-by-side comparison images.

    Returns a dict with the numeric read-out plus data-URI images:
        original, annotated, leaf_mask, lesion_mask
    """
    sev = _load_severity()

    # The rebuilt severity.py exports analyse_leaf() + assess_severity() and no longer
    # has severity_from_image(). Both call shapes are accepted so an older artifacts
    # folder still works, but the masks come from analyse_leaf either way - the numbers
    # and the pictures must describe the same segmentation.
    analysis = None
    if hasattr(sev, "analyse_leaf") and hasattr(sev, "assess_severity"):
        analysis = sev.analyse_leaf(bgr)
        result = dict(sev.assess_severity(bgr, analysis))
        result["lesion_mask"] = analysis.get("lesion_mask")
        result["leaf_mask"] = analysis.get("leaf_mask")
        result["boxes"] = analysis.get("boxes")
    elif hasattr(sev, "severity_from_image"):
        try:
            result = sev.severity_from_image(bgr, crop=crop)
        except TypeError:                  # older severity.py without the crop arg
            result = sev.severity_from_image(bgr)
    else:
        raise AttributeError(
            "artifacts/severity.py exports neither analyse_leaf/assess_severity nor "
            "severity_from_image. It is not a module this app recognises - re-export "
            "it from notebook 01.")

    lesion = result.get("lesion_mask")
    leaf = result.get("leaf_mask")
    annotated = overlay_lesions(bgr, lesion, result.get("boxes"))

    pct = result.get("severity_percent")
    return {
        "severity_level": result.get("severity_level", "Unavailable"),
        "severity_percent": pct,
        "severity_pct_display": "-" if pct is None else f"{pct:.1f}%",
        "lesion_count": int(result.get("lesion_count", 0) or 0),
        "largest_lesion_pct": result.get("largest_lesion_pct"),
        "leaf_fraction": result.get("leaf_fraction"),
        "mask_ok": bool(result.get("mask_ok", False)),
        "images": {
            "original": to_data_uri(bgr),
            "annotated": to_data_uri(annotated),
            "leaf_mask": to_data_uri(_mask_to_bgr(leaf, bgr.shape)) if leaf is not None else None,
            "lesion_mask": to_data_uri(_mask_to_bgr(lesion, bgr.shape)) if lesion is not None else None,
        },
    }


def read_upload(file_storage):
    """Werkzeug FileStorage -> BGR array, without touching disk."""
    data = np.frombuffer(file_storage.read(), np.uint8)
    file_storage.seek(0)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode that file as an image.")
    return bgr
