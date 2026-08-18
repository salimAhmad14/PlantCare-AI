# What changed in this update

Same UI. `templates/` and `static/style.css` keep their existing structure; the new
content slots into the `finding` sections and the `plan` block, plus two new sections
(`advisory`, `differential`) styled with the same vocabulary.

---

## 1. Every artifact was stale — all replaced

The previous zip was still serving the **27-class wheat build**:

| Artifact | Was | Now |
|---|---|---|
| `class_index.json` | 27 classes incl. wheat | 13 |
| `best_cnn.pt` | 27-class head | 13-class, val acc 0.9988 |
| `severity.py` | pre-rebuild ExG, bands 1.5/10/25 | chroma-split rebuild, bands 5/15/35 |
| `chunks.csv`, `kb_manifest.json` | 214 chunks / 25 diseases | 172 chunks / 11 documents |
| `plantcare.db` | built from those 214 | rebuilt (`rebuild_db.py`) |
| `data/rag_documents/` | 11 wheat PDFs | removed |

Newly added, absent before: `run_config.json`, `ood_gaussian.npz`, `preprocessing.py`,
`kb_chunks.json`.

---

## 2. Bugs found while wiring it up

**`Path(__file__)` is relative** when you run `python app.py`, so `.parent` and
`.parent.parent` are both `"."` and the project-root `.env` was never found — the app
died on a missing secret key. Now `.resolve()`d, and loaded from `config.py` rather than
`app.py`, because config is the module everything imports first.

**The embedder key was misread.** Notebook 02 writes `embed_model`; the app read
`embedding_model` only, so it silently fell back to **bge-small (384-d)** against
**768-d** vectors. Both spellings are now accepted and the dimension is asserted against
`embeddings.npy` at startup.

**The Milvus hybrid probe hardcoded `dim = 384`**, so a 768-d collection failed the probe
every time and dropped to dense-only while looking healthy. It now reads the dimension
from the schema.

**Column vocabulary drift.** The rewritten notebook 02 emits `disease`/`page`/`doc`; the
app expected `all_diseases`/`page_no`/`source_pdf`. Every query raised `KeyError`.
Normalised once in `kb_backends.normalise_columns()`.

**`severity.py`'s API changed.** It no longer has `severity_from_image()`; it exports
`analyse_leaf()` + `assess_severity()`. `leaf_analysis.py` accepts both.

**`_expr` was a `@staticmethod` referencing `self`** after the disease-field fix — 500 on
every Milvus query.

**`load_milvus.py` called `config.COLLECTION`**, which `config.py` never defined.

---

## 3. Missing safety, now present

**Refusal screens.** The app had a confidence gate only. A 13-class softmax always sums
to 1, so an unseen species produced a confident, cited, wrong diagnosis. Notebook 01's
other two screens are now wired in: leaf-presence (`leaf_fraction` outside 0.06–0.97) and
Mahalanobis distance in the 1280-d penultimate space against threshold 669.57.

They run **before** the healthy branch. That ordering matters: a photo of a hand was
being classified as a healthy leaf and returned as "no disease detected" — a false
all-clear. A refused image never reaches the knowledge base at all.

**Preprocessing at inference.** Notebook 01 trains *and* serves with `preprocess_image()`;
the app skipped it. That is a train/serve mismatch — the model saw a different input
distribution from the one it was validated on. Now applied, and `/health` reports whether
it is on.

**Severity conflict.** The lesion mask is chroma/darkness based: it measures brown
necrosis well and *diffuse* symptoms badly. A correctly classified leaf-mold image
measured 0.1% and landed in "Healthy" — and the app printed *"No action needed."* over a
confirmed disease. A diseased leaf whose band is `Healthy` or `Unavailable` now has its
severity treated as **unmeasured, not as an all-clear**, with urgency floored at
"act within the week".

> The classifier decides **whether** there is disease. The mask only decides **how much**.

---

## 4. New capability

**Field advisory (`advisor.py`).** Groq by default, OpenAI-compatible, model discovered
at runtime because free-tier catalogues change without notice. The prose sits **above**
the extractive passages, never instead of them — with no API key the report is exactly
what it was before.

Three fences: closed-book prompt, mandatory `[S#]` citations, and a post-generation audit
that checks every number-with-a-unit, every chemical name and every citation index
against the source text. Anything unsupported and the whole draft is discarded.

The audit also greps for **image claims**. The model has not seen the photograph, so
*"the leaf shows brown concentric rings"* is invented — and it would read as the most
authoritative line in the report.

**Look-alike differential (`differential.py`).** Derived, not hand-written: a symptom
centroid per class, nearest neighbours within the same crop. On this KB that gives
early blight ↔ target spot at 0.928 (both "concentric rings" — the classic tomato
misdiagnosis), septoria ↔ bacterial spot, spider mites ↔ leaf mold at 0.794 (weakest,
correctly — webbing looks like nothing else).

The three grape classes all sit near 0.89 because each is compared against only two
others, so the grape differential is inherently less discriminating than tomato.

**Prevention is its own block.** `SECTION_QUERIES` had four blocks with `precaution`
mapped to `treatment`, so prevention passages were never retrieved despite being 34 of
the 172 chunks.

**`/health` fingerprint check.** Reports the screens, the advisory status and the
differential, and fails if `class_index.json` and `kb_manifest.json` describe different
class sets. That is exactly what would have caught the stale-artifact mismatch on day one.

---

## 5. Retrieval backend default changed to `numpy`

Milvus Lite has no server-side BM25 Function, so its hybrid probe fails and it runs
**dense-only**. The numpy backend computes BM25 itself and fuses via RRF, so on this
corpus it is strictly the better retriever, and at 172 chunks the speed difference is
irrelevant.

`plantcare.db` is still shipped and works — set `PLANTCARE_KB_BACKEND=milvus` to use it,
or point `PLANTCARE_MILVUS_URI` at a real server, which does support BM25.

---

## Before you run

```bash
pip install -r app/requirements.txt
```

Then put your keys in `.env` (it is gitignored now; it was not before):

```
GROQ_API_KEY=...      # free, no card: https://console.groq.com
HF_TOKEN=...          # optional, avoids rate limits on the model download
```

```bash
cd app && python app.py          # http://127.0.0.1:5000
curl http://127.0.0.1:5000/health
```

`/health` should show `"ok": true`, no `missing`, and `advisory.enabled: true` once the
key is in place.

**Rotate any key that has ever been pasted into a chat, an issue or a screenshot.**

---

## 6. Tested against your own `data/testing_images/` — read this part

Four real photographs, run through the finished app. Classification is real (only the
sentence-transformers encoder was stubbed in my sandbox, which does not touch the CNN):

| Image | Predicted | Conf | Mahalanobis | Outcome |
|---|---|---|---|---|
| `Grape_healthy_leaf.jpg` | grape_healthy | 95.0% | 68 | **correct**, published |
| `Septoria_Leaf_Spot_of_Tomato2241.webp` | grape_black_rot | 95.1% | 65 | **wrong**, refused (frame filled) |
| `grape_healhty_dark.webp` | tomato_late_blight | 73.1% | 283 | **wrong**, refused (low confidence) |
| `tomato_bacterial_spot.webp` | grape_black_rot | 95.7% | 56 | **wrong, and published** |

Three of four are misclassified, and two of those are wrong with 95%+ confidence.

**The OOD screen does not catch them.** The threshold is 669.57; these images score
56–283, deep inside the in-distribution range. They *look* like training data in the
1280-d feature space while being wrong — which is the failure mode Mahalanobis is
supposed to catch and here does not.

This is not a bug in the app. It is dataset shift. The 0.9986 test accuracy was measured
on a held-out split of the same PlantVillage-style corpus: uniform backgrounds, similar
lighting, similar framing. Web photographs differ on all three. The note already in the
project — that ~14% of the dataset is byte-identical duplicates still inside train — makes
the headline number softer still.

Worth doing before you demo or write this up:

1. Run notebook 03's batch cell over a proper sample of the **held-out test split** and
   quote that number, not 0.9986.
2. Collect 20–30 web images per crop, label them, and measure accuracy separately. That
   second number is the one that describes deployment.
3. If the gap is large, the fix is training data, not a threshold: web-like images with
   varied backgrounds, or background augmentation.
4. Re-calibrate `OOD_THRESHOLD` against those web images rather than against val. It was
   set at `OOD_RETENTION=0.995` on validation data, which is why it passes them.

The honest framing for your report: the classifier is strong **within its training
distribution** and the refusal machinery is what stands between it and a confident wrong
answer outside it. Two of the three errors above were caught. One was not.

## 7. Two more bugs the real images found

**A healthy leaf was being told to act.** `Grape_healthy_leaf.jpg` classified healthy at
95%, but the mask read 8.1% non-green (leaf edge, vein shadow) and the report printed
*"low — act within the week."* On a leaf the classifier called healthy, a few percent of
non-green is noise, not disease. The healthy branch now forces monitoring-only urgency.

**"No leaf detected" on a leaf filling the whole frame.** The septoria macro shot has
`leaf_fraction = 0.994`, above `VEGETATION_MAX = 0.97`, and was refused with
*"no leaf detected"* — the opposite of what is wrong. Split into its own `frame_filled`
reason: *"the leaf fills the entire frame"*, with the actionable instruction to step back
so some background is visible. The threshold itself is unchanged; it is notebook 01's,
and it is doing its job — the segmenter genuinely has no background to measure against.

---

## 6. A finding from the bundled test images — read this

The four images in `data/testing_images/` were run through the finished pipeline. Only
one was classified correctly:

| Image | Predicted | Confidence | OOD distance | Screen |
|---|---|---|---|---|
| `Grape_healthy_leaf.jpg` | `grape_healthy` | 95.0% | 67.7 | accept — **correct** |
| `tomato_bacterial_spot.webp` | `grape_black_rot` | 95.7% | 55.7 | accept — **wrong** |
| `Septoria_Leaf_Spot_of_Tomato2241.webp` | `grape_black_rot` | 95.1% | 65.1 | refused (leaf fills 99% of frame) |
| `grape_healhty_dark.webp` | `tomato_late_blight` | 73.1% | 283.1 | refused (low confidence) |

This is **not** an app bug. Ruled out by testing: the class order matches the checkpoint's
own `class_to_id` exactly, and turning preprocessing off changes nothing (95.7% -> 96.4%,
same wrong label).

It is a **distribution gap**. These are web photographs — plants in the field, varied
scale, varied background. The training set is single leaves on uniform backgrounds at
consistent scale. The model has never seen this kind of picture.

The uncomfortable part is that the **Mahalanobis screen did not catch it**: distances of
55–65 against a threshold of 669.57. That threshold was calibrated on the validation
split at 0.995 retention — i.e. tuned to keep 99.5% of images that look *like the training
data*. It is therefore loose by construction. It will catch a photo of a hand or a
keyboard. It will not catch a real, correctly-cropped leaf that was simply photographed
differently.

What this means in practice:

- The 99.86% test accuracy is on held-out data from the **same** distribution, and
  ~14% of the dataset is still byte-identical duplicates inside train. Treat it as an
  upper bound, not a field estimate.
- Two of four real photographs were refused, which is the system behaving conservatively
  and is the right outcome. One passed with a confident wrong answer, which is the
  failure mode the screens exist to prevent and did not.

Worth doing before any field claim: collect 30–50 phone photographs of grape and tomato
leaves in real conditions, run them through `/analyse`, and record accuracy and refusal
rate separately. That number — not the 99.86% — is what belongs in the report. If it is
poor, the fix is training data from that distribution, or a much tighter OOD threshold
accepting a higher refusal rate.

---

## 7. Report trimmed and front page corrected

**Front page.** The eyebrow said "Grape · Tomato · Wheat"; wheat is gone. Step 1 claimed
41,713 images and 27 classes — recomputed from `manifest.csv`, the real figures are
**28,843 images across 13 classes** (train 20,833 / val 4,330 / test 3,680; 24,763 unique
by MD5, so 4,080 duplicates). Step 2 also still described the old excess-green-only mask,
which the severity rebuild replaced.

**Report page.** Everything below "Save this report" is removed: the Symptoms, Cause,
Why it spread, Treatment and Prevention blocks, the look-alike differential, and the
standalone Sources list. The page now ends at the download buttons.

**One safeguard kept.** The removed blocks reappear **only when no written advisory was
produced** — no API key, API down, or the audit rejected the draft. Without that the
report would be empty in exactly the situations where something has gone wrong, which is
the worst moment to show a blank page. When the advisory renders, none of them appear.

The advisory's own collapsed "Which source is which" list stays. The inline `[S1]`
markers are the mechanism the audit verifies against, so they need something to resolve
to; it is one closed `<details>` line and can be deleted from `result.html` if you would
rather it went.

PDF, Word and plain-text downloads follow the same rule, so a saved report matches what
was on screen.
