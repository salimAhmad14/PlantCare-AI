# PlantCare-AI

An end-to-end plant leaf disease diagnosis system for **grape and tomato** crops. Upload a leaf photo and the system identifies the disease, estimates how badly the leaf is affected, and generates a treatment advisory where every recommendation is traced back to a source document.

Built as a Postgraduate Certificate project at C-DAC Noida.

---

## What it does

```
Leaf image
   |
   v
[1] Image preprocessing        deblur + brightness correction
   |
   v
[2] CNN classification         EfficientNet-B0, 13 disease classes
   |
   v
[3] Rejection screens          refuses non-leaf / unknown / low-confidence images
   |
   v
[4] Lesion segmentation        measures % of leaf tissue affected
   |
   v
[5] Knowledge retrieval        hybrid search over a curated disease knowledge base
   |
   v
[6] Audited advisory           LLM report with mandatory [S#] citations
   |
   v
Web report + PDF / Word / text download
```

---

## Key features

**Refuses instead of guessing.** A 13-class softmax always sums to 1, so an unseen plant species would still get a confident wrong label. Three independent screens prevent this:

1. **Leaf presence** — checks that enough green tissue is actually in the frame
2. **Mahalanobis distance** — measures how far the image sits from the training distribution in the model's 1280-dimensional feature space
3. **Confidence gate** — rejects predictions below 0.80

**Severity is measured, not guessed.** The leaf-vs-background decision and the healthy-vs-diseased decision are handled by two separate steps. A single green-based threshold used to delete brown necrotic tissue from the leaf entirely, removing it from both the numerator and the denominator, which reported margin diseases as 0% severe. Separating the two jobs fixed it.

**The classifier decides *whether*, the mask decides *how much*.** If the CNN names a disease but the mask measures near-zero severity (common with diffuse symptoms like leaf mold or viral chlorosis), the system treats severity as **unmeasured** rather than as good news, and floors the urgency instead of telling the farmer "no action needed."

**No invented chemistry.** Every generated advisory is checked after generation: numbers with units, chemical names, and citation markers must all appear in the retrieved source text. If the check fails, the system falls back to an extractive summary where every line is verbatim from a source passage. A hallucinated fungicide dosage that reads as verified would be worse than no answer.

**Data leakage was audited, not assumed.** An MD5 + perceptual-hash audit found 1,820 duplicate groups straddling the train/val/test splits — 12.65% of the dataset. All duplicate groups were moved into a single split before any training happened.

---

## Results

| Metric | Value |
|---|---|
| Classes | 13 (4 grape, 9 tomato) |
| Images | 28,843 |
| Architecture | EfficientNet-B0, 4.04M parameters |
| Test accuracy | 99.86% |
| Macro F1 | 0.9984 |
| Weakest class | tomato_late_blight (0.992) |
| Inference | ~3.6 ms/image |
| Knowledge base | 172 chunks across 11 disease documents |

### Model comparison

| Model | Params | Test Acc | Macro F1 | Size |
|---|---|---|---|---|
| **EfficientNet-B0** | 4.04M | 0.9562 | 0.9200 | 16 MB |
| DenseNet-121 | 6.98M | 0.9537 | 0.9155 | 28 MB |
| Faster R-CNN R50-FPN | 40.72M | 0.8140 | 0.7368 | 160 MB |

*Benchmark run under a shorter, shared training budget — a lower bound, not a tuned comparison.*

---

## Supported classes

**Grape** — black measles, black rot, isariopsis leaf spot, healthy

**Tomato** — bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, spider mites, target spot, yellow leaf curl virus, healthy

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Classification | PyTorch, EfficientNet-B0 | best accuracy-per-parameter for CPU deployment |
| Segmentation | OpenCV (Lab chroma + HSV) | interpretable, no extra labelled masks needed |
| Embeddings | BAAI/bge-base-en-v1.5 (768-d) | asymmetric query/passage instructions suit short machine-built queries |
| Keyword search | BM25, fused with dense via RRF | RRF fuses ranks, so no scale calibration needed |
| Vector store | NumPy backend (default) / Milvus Lite | at 172 chunks NumPy computes BM25 itself; Milvus Lite is dense-only |
| Generation | Groq API (Llama 3.3 70B) | free tier, no local GPU required |
| Web app | Flask + Jinja | server-rendered, no JS build step |
| Reports | ReportLab, python-docx | PDF and Word export |

---

## Repository layout

```
app/
  app.py             Flask routes
  classifier.py      CNN inference + rejection screens
  leaf_analysis.py   lesion segmentation and severity
  knowledge_base.py  retrieval
  kb_backends.py     NumPy / Milvus Lite / Milvus server
  advisor.py         LLM generation + audit
  differential.py    look-alike disease suggestions
  action_plan.py     extractive step ranking
  export.py          PDF / Word / text export
  artifacts/         model weights, config, knowledge base
  templates/
notebook/
  Notebook 01        classification pipeline
  Notebook 02        knowledge base construction
  Notebook 03        image-to-advisory pipeline
```

---

## Setup

```bash
git clone https://github.com/salimAhmad14/PlantCare-AI.git
cd PlantCare-AI

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then add your GROQ_API_KEY
```

Run it:

```bash
cd app
python app.py
```

Open `http://localhost:5000`. Check `http://localhost:5000/health` to confirm the model, knowledge base and class index all agree with each other.

### Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | LLM advisory generation (falls back to extractive without it) |
| `PLANTCARE_KB_BACKEND` | `numpy` (default) or `milvus` |
| `PLANTCARE_ARTIFACT_DIR` | point at an alternative artifacts folder |
| `PLANTCARE_CORS_ORIGIN` | allowed origin for the JSON API |

`.env` is gitignored. Never commit it.

---

## Honest limitations

Worth reading before you trust a number from this repo.

- **Field photos are the weak point.** The 99.86% test accuracy is measured on single leaves against uniform backgrounds. On real-world web-sourced field photos, accuracy drops sharply and the out-of-distribution screen does not reliably catch the failures — the calibrated threshold is loose by construction. Anyone deploying this should collect 30–50 real phone photos and measure accuracy *and* refusal rate on those.
- **~14% of images are byte-identical duplicates** still inside the training split. The leakage audit only repaired cross-split duplicates, so the effective unique dataset is closer to 24,763 images.
- **Severity bands are uncalibrated.** The 5 / 15 / 35 percent cutoffs are reasonable estimates, not values validated against agronomist-labelled ground truth.
- **The grape differential is structurally weaker.** With only 3 grape disease classes, the nearest-neighbour look-alikes are less discriminating than tomato's.
- **The mask measures necrosis, not chlorosis.** Diffuse symptoms are systematically under-measured; this is why the urgency floor exists.
- **Advisory output is decision support, not a prescription.** Always confirm pesticide choice and dosage with a local agricultural extension officer.

---

## Roadmap

- Grad-CAM explanation overlays
- Field-photo dataset collection and retraining
- Severity calibration against expert-labelled samples
- More crops beyond grape and tomato
- Mobile-friendly capture flow

---

## Team

Salim Ahmad · Ankush Napit · Mohit Verma

C-DAC Noida — PG Certificate Program in Artificial Intelligence

---

## License

MIT
