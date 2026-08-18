"""Which diseases look like this one, derived rather than hand-written.

A curated confusion list would be an opinion. Instead this computes a SYMPTOM CENTROID
per class - the mean of that class's `symptom` chunk embeddings - and takes the nearest
other classes in the same crop by cosine similarity.

That is defensible because it measures what the corpus actually says: if two diseases
are described with similar symptom language, a farmer looking at a leaf will have the
same trouble the embedding does. It also updates itself when the documents change.

Measured on the current knowledge base with bge-base:

    tomato early blight  <-> target spot      0.928   (both "concentric rings")
    tomato bacterial spot<-> late blight      0.906
    tomato septoria      <-> early blight     0.900
    tomato spider mites  <-> leaf mold        0.794   (weakest - webbing looks like
                                                       nothing else, which is correct)

The three grape classes all sit near 0.89 because there are only three of them, so each
is compared against just two others; the grape differential is inherently less
discriminating than tomato. Worth knowing before reading too much into it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config

_centroids = None      # {label: unit vector}
_loaded = False


def _load():
    global _centroids, _loaded
    if _loaded:
        return _centroids
    _loaded = True

    chunks_p, vecs_p = Path(config.KB_CHUNKS), Path(config.KB_VECTORS)
    if not (chunks_p.exists() and vecs_p.exists()):
        print("[differential] chunks.csv / embeddings.npy missing - differential off")
        return None

    df = pd.read_csv(chunks_p)
    vecs = np.load(vecs_p)
    if len(df) != len(vecs):
        print(f"[differential] chunks.csv has {len(df)} rows but embeddings.npy has "
              f"{len(vecs)} - refusing to align them by position. Differential off.")
        return None

    if "section_type" not in df or "disease" not in df:
        print("[differential] chunks.csv lacks section_type/disease - differential off")
        return None

    out = {}
    sym = df.index[(df["section_type"] == "symptom") & df["disease"].notna()]
    for label, idx in df.loc[sym].groupby("disease").groups.items():
        if "healthy" in str(label):
            continue
        v = vecs[np.asarray(idx, dtype=int)].mean(axis=0)
        n = float(np.linalg.norm(v))
        if n > 0:
            out[str(label)] = v / n

    _centroids = out or None
    if _centroids:
        print(f"[differential] symptom centroids for {len(_centroids)} classes")
    return _centroids


def look_alikes(label, k=None):
    """-> [(other_label, similarity), ...] within the same crop, most similar first."""
    if not config.DIFFERENTIAL_ENABLED:
        return []
    cents = _load()
    if not cents or label not in cents:
        return []
    k = config.DIFFERENTIAL_K if k is None else k
    crop = str(label).split("_")[0]
    sims = [(float(cents[label] @ cents[other]), other)
            for other in cents
            if other != label and str(other).split("_")[0] == crop]
    sims.sort(reverse=True)
    return [(other, sim) for sim, other in sims[:k]]


def health():
    cents = _load()
    return {"enabled": bool(config.DIFFERENTIAL_ENABLED),
            "classes_with_centroids": len(cents) if cents else 0,
            "k": config.DIFFERENTIAL_K}
