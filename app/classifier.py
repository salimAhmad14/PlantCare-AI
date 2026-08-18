"""EfficientNet-B0 classifier wrapper.

Loaded once on first use and kept in memory. CPU is fine - a single 224px forward
pass is a few tens of milliseconds.
"""

import importlib.util
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

import config

_model = None
_tf = None
_classes = None
_gate = config.CONFIDENCE_GATE
_warnings = []
_preprocess = None          # notebook 01's preprocess_image, or None
_ood = None                 # (class_means, precision, threshold), or None


def _load_preprocess():
    """Import notebook 01's preprocessing.py rather than reimplementing it.

    Notebook 01 applies this to all three splits AND at inference, so the app must
    apply it too - the transforms are fixed per-image (deblur gate, CLAHE, gamma),
    not fitted, so there is no leakage, but skipping them at serve time IS a
    train/serve mismatch.
    """
    global _preprocess
    if not config.APPLY_PREPROCESSING:
        return None
    if not config.PREPROCESS_MODULE.exists():
        _warnings.append(
            f"{config.PREPROCESS_MODULE.name} missing - images are NOT preprocessed, "
            "but notebook 01 trained and validated with preprocessing on. Copy it "
            "from the notebook 01 artifacts.")
        print("[classifier] WARNING", _warnings[-1])
        return None
    spec = importlib.util.spec_from_file_location("nb01_preprocessing",
                                                  config.PREPROCESS_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _preprocess = mod.preprocess_image
    return _preprocess


def _load_ood():
    """Class means + shared precision from notebook 01, for the Mahalanobis screen."""
    global _ood
    if not config.OOD_GAUSSIAN.exists():
        _warnings.append(
            f"{config.OOD_GAUSSIAN.name} missing - the unknown-leaf screen is OFF. A "
            "13-class softmax always sums to 1, so a photo of an unseen species will "
            "get a confident, wrong, fully-cited diagnosis.")
        print("[classifier] WARNING", _warnings[-1])
        return None
    z = np.load(config.OOD_GAUSSIAN)
    thr = config.OOD_THRESHOLD
    if thr is None:
        thr = float(z["threshold"][0]) if "threshold" in z else None
    _ood = (z["class_means"], z["precision"], thr)
    return _ood


def mahalanobis(features, means, precision):
    """Distance to the NEAREST class centre. Small = looks like the training data."""
    best = None
    for centre in means:
        d = features - centre
        dist = np.einsum("ij,jk,ik->i", d, precision, d)
        best = dist if best is None else np.minimum(best, dist)
    return np.sqrt(np.maximum(best, 0.0))


def _build(num_classes):
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.classifier[1].in_features, num_classes))
    return m


def load():
    global _model, _tf, _classes, _gate
    if _model is not None:
        return

    if not config.CNN_WEIGHTS.exists():
        raise FileNotFoundError(
            f"{config.CNN_WEIGHTS} missing - copy best_cnn.pt from Kaggle into artifacts/")
    if not config.CLASS_INDEX.exists():
        raise FileNotFoundError(f"{config.CLASS_INDEX} missing - copy class_index.json too")

    raw = json.loads(config.CLASS_INDEX.read_text())
    if isinstance(raw, dict):
        if all(str(k).lstrip("-").isdigit() for k in raw):          # {"0": "grape_..."}
            _classes = [raw[k] for k in sorted(raw, key=lambda x: int(x))]
        else:                                                        # {"grape_...": 0}
            _classes = [k for k, _ in sorted(raw.items(), key=lambda kv: kv[1])]
    else:
        _classes = list(raw)

    ckpt = torch.load(config.CNN_WEIGHTS, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    if not cfg:
        _warnings.append(
            "Checkpoint has no 'config' block, so image size / normalisation come from "
            "config.py. If notebook 01 trained with different values, this app "
            "preprocesses differently from training and nothing will flag it. Add a "
            "config= dict to the torch.save in notebook 01.")
        print("[classifier] WARNING", _warnings[-1])

    size = int(cfg.get("img_size", config.IMG_SIZE))
    mean = cfg.get("mean", config.MEAN)
    std = cfg.get("std", config.STD)
    _gate = float(cfg.get("confidence_gate", config.CONFIDENCE_GATE))

    _model = _build(len(_classes))

    # strict=True on purpose. With strict=False a shape or naming mismatch loads a
    # randomly-initialised head and the app serves confident nonsense in silence.
    try:
        _model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        missing = sorted(set(_model.state_dict()) - set(state))[:6]
        extra = sorted(set(state) - set(_model.state_dict()))[:6]
        raise RuntimeError(
            f"Checkpoint does not match the model definition.\n"
            f"  missing in checkpoint : {missing}\n"
            f"  unexpected in checkpoint: {extra}\n"
            f"  class_index.json has {len(_classes)} classes\n"
            f"Original error: {exc}") from exc

    _model.eval()
    _load_preprocess()
    _load_ood()
    _tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    thr = _ood[2] if _ood else None
    print(f"[classifier] {len(_classes)} classes | img {size} | conf gate {_gate:.0%} | "
          f"preprocess {'on' if _preprocess else 'OFF'} | "
          f"OOD {'threshold %.1f' % thr if thr else 'OFF'}")


def class_names():
    load()
    return list(_classes)


def confidence_gate():
    load()
    return _gate


def warnings():
    return list(_warnings)


@torch.no_grad()
def predict(bgr, topk=config.TOP_K):
    """Returns (label, confidence, [(label, prob), ...])."""
    label, conf, top, _ = predict_full(bgr, topk)
    return label, conf, top


@torch.no_grad()
def predict_full(bgr, topk=config.TOP_K):
    """Returns (label, confidence, top-k, extras) where extras carries the OOD distance.

    Mirrors notebook 01's screen_and_classify: preprocess -> eval_tf -> penultimate
    features -> logits, with the Mahalanobis distance taken from the same forward pass
    rather than a second one.
    """
    load()
    prepared = _preprocess(bgr) if _preprocess else bgr
    x = _tf(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)).unsqueeze(0)

    pooled = torch.flatten(_model.avgpool(_model.features(x)), 1)
    probs = torch.softmax(_model.classifier(pooled), 1)[0].numpy()

    extras = {"ood_distance": None, "ood_threshold": None, "preprocessed": bool(_preprocess)}
    if _ood is not None:
        means, precision, thr = _ood
        extras["ood_distance"] = float(mahalanobis(pooled.numpy(), means, precision)[0])
        extras["ood_threshold"] = thr

    order = probs.argsort()[::-1][:topk]
    top = [(_classes[int(i)], float(probs[int(i)])) for i in order]
    return top[0][0], top[0][1], top, extras


def screen(leaf_fraction, confidence, ood_distance):
    """The three refusal screens, in notebook 01's order.

    A confidence gate alone is not enough: softmax over 13 classes always sums to 1,
    so an unseen species produces a CONFIDENT wrong answer. The leaf-presence and
    Mahalanobis screens are what actually catch it.

    -> (reason, detail) or (None, None) when the image is accepted.
    """
    load()
    if leaf_fraction is not None:
        if leaf_fraction < config.VEGETATION_MIN:
            return "no_leaf", f"leaf covers only {leaf_fraction:.1%} of the frame"
        if leaf_fraction > config.VEGETATION_MAX:
            # Same screen, different cause, and the honest message is the opposite of
            # "no leaf detected": a macro shot where the leaf fills the frame leaves the
            # segmenter no background to measure against, so severity is unusable even
            # though there is plainly a leaf. Telling the user to step back is
            # actionable; telling them no leaf was found is confusing and wrong.
            return "frame_filled", (f"leaf fills {leaf_fraction:.1%} of the frame, "
                                    f"leaving no background to measure against")

    if _ood is not None and _ood[2] is not None and ood_distance is not None:
        if ood_distance > _ood[2]:
            return "unknown_leaf", f"distance {ood_distance:.1f} / threshold {_ood[2]:.1f}"

    if confidence < _gate:
        return "low_confidence", f"confidence {confidence:.2f} / threshold {_gate:.2f}"
    return None, None


def pretty(label):
    """grape_black_rot -> ("Grape", "Black Rot", "Grape Black Rot")."""
    parts = str(label).split("_")
    crop = parts[0].title()
    disease = " ".join(parts[1:]).title() if len(parts) > 1 else ""
    return crop, disease, f"{crop} {disease}".strip()
