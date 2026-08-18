# PlantCare — Flask advisory app

Upload a leaf photograph → EfficientNet-B0 names the disease → the lesion mask
measures severity → tiered retrieval pulls cited IPM text → a report you can read
or download.

---

## Quick start (Windows / PowerShell)

```powershell
cd S:\PlantCare\plantcare_web\plantcare_web
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

$env:PLANTCARE_SECRET_KEY = -join ((48..57)+(97..122) | Get-Random -Count 40 | % {[char]$_})
.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000

Calling `.venv\Scripts\python.exe` directly avoids `Activate.ps1`, which PowerShell
blocks by default (`SecurityError: UnauthorizedAccess`). If you would rather activate:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

`set VAR=value` is cmd.exe syntax and silently does nothing in PowerShell — use
`$env:VAR="value"`.

macOS / Linux: `source .venv/bin/activate` and `export`.

Check everything loaded before uploading anything:

```
http://127.0.0.1:5000/health
```

It reports which backend the knowledge base chose, how many chunks it holds, and
whether hybrid search is actually running.

---

## Backends

`knowledge_base.py` picks one automatically. You do not have to configure anything.

| order | condition | backend |
|---|---|---|
| 1 | `PLANTCARE_MILVUS_URI` set | Milvus **server** |
| 2 | `artifacts/plantcare.db` present and `pymilvus` importable | Milvus **Lite** |
| 3 | `artifacts/chunks.csv` + `embeddings.npy` | **numpy** |

**On Windows you will land on numpy**, because `milvus-lite` publishes no Windows
wheel. That is expected and fully supported — the numpy backend carries its own
BM25 implementation, so dense + lexical fusion works there too. Force a backend
with `PLANTCARE_KB_BACKEND=numpy|milvus` if you need to compare them.

To run the Milvus server instead (needs Docker/WSL2):

```bash
docker compose up -d
python load_milvus.py
export PLANTCARE_MILVUS_URI=http://localhost:19530
python app.py
```

---

## Environment variables

| variable | default | meaning |
|---|---|---|
| `PLANTCARE_DEBUG` | off | `1` enables Flask debug pages and a throwaway secret key |
| `PLANTCARE_SECRET_KEY` | — | **required** when debug is off; the app refuses to start without it |
| `PLANTCARE_CORS_ORIGIN` | empty | set to your frontend origin to allow `POST /api/analyse` |
| `PLANTCARE_MILVUS_URI` | empty | Milvus server URI |
| `PLANTCARE_KB_BACKEND` | `auto` | `numpy` or `milvus` to force one |
| `PLANTCARE_HOST` / `PLANTCARE_PORT` | `127.0.0.1` / `5000` | bind address |

---

## After rebuilding the knowledge base

Copy these five out of `plantcare_kb.zip` into `artifacts/`, replacing what is there:

```
chunks.csv   embeddings.npy   plantcare.db/   kb_manifest.json   knowledge_graph.json
```

Then restart and read the first four lines of console output:

```
[kb] backend    : milvus | dense + BM25 with RRF
[kb] chunks     : 312 from 25 documents
[kb] vocabulary : v2 (seven sections)
[kb] sections   : identity, overview, pathogen, prevention, symptom, transmission, treatment
```

If it says `LEGACY (four sections)` the old KB is still in place — `cause` and `reason`
will share one pool and one of them will render empty. The banner exists because the
commonest failure after a rebuild is forgetting to copy the artifacts and then concluding
the rebuild changed nothing.

Optionally, point at a KB export elsewhere on disk without copying anything. Use a real
path — a directory that does not exist now warns and falls back to `artifacts/`:

```powershell
$env:PLANTCARE_ARTIFACT_DIR="S:\PlantCare\kb_export"
python app.py

# clear it again when you are done
Remove-Item Env:\PLANTCARE_ARTIFACT_DIR
```

## Artifacts

Everything in `artifacts/` comes out of the Kaggle notebooks:

| file | from | used for |
|---|---|---|
| `best_cnn.pt` | notebook 01 | classifier weights |
| `class_index.json` | notebook 01 | class order |
| `severity.py` | notebook 01 | lesion mask + severity bands |
| `chunks.csv`, `embeddings.npy` | notebook 02 | numpy backend; BM25 indexes `search_text` |
| `plantcare.db/` | notebook 02 | Milvus Lite backend |
| `kb_manifest.json` | notebook 02 | collection name, embedding model, thin/missing classes |
| `knowledge_graph.json` | notebook 02 | BM25 query expansion |

`severity.py` lives **only** in `artifacts/`. The previous build kept a second copy
at the project root; they were identical at the time and would have drifted.
`leaf_analysis.py` imports the artifacts copy so the app and the notebook can never
disagree about a severity number.

---

## Routes

| route | method | purpose |
|---|---|---|
| `/` | GET | upload form |
| `/analyse` | POST | HTML report |
| `/api/analyse` | POST | JSON report (`?images=1` to include base64 images) |
| `/download` | POST | plain-text report |
| `/health` | GET | artifact + backend readiness |

---

## What changed from the previous build

| | problem | fix |
|---|---|---|
| 1 | `app.py` had a duplicated `if __name__ == "__main__":` and did not compile at all | removed |
| 2 | `knowledge_base.py` was missing `to_passages()` and `fallback_classes()`, which `report.py` calls — `AttributeError` on the first request | both implemented |
| 3 | the numpy backend had been dropped, and `MilvusClient` was constructed at import time — the module could not even import on Windows | `kb_backends.py` provides both, selected at runtime |
| 4 | `flask` was not in `requirements.txt` | added, with `waitress`/`gunicorn` per platform |
| 5 | two Windows virtualenvs shipped inside the zip (~25 MB of 26 MB) | `.gitignore`, no venvs |
| 6 | `load_state_dict(..., strict=False)` silently accepted a mismatched checkpoint and would serve a random head | `strict=True`, with the key diff in the error |
| 7 | `CORS_ORIGIN = "*"` and a hardcoded secret key | both env-driven; no CORS header unless configured; startup fails if the key is unset outside debug |
| 8 | disease filter used SQL `like "%x%"` — substring matching on a comma-joined field | exact token match, applied in Python on both backends |
| 9 | graph expansion was fed to the dense encoder, turning a sentence into a bag of words | expansion goes to BM25 only; dense gets the original query |
| 10 | "Cause" and "Reason" both query `section_type="biology"` and printed the same passage twice | cross-block dedup; an empty second block is reported honestly |
| 11 | a healthy diagnosis could print "40.3% of leaf area — High: act immediately" | `_reconcile()` treats that contradiction as a mask failure, withholds the number, states the disagreement |

---

## Known limits

**Content-rescued passages.** Notebook 02 fills a missing section from chunk content when
the source document has no heading for it. Those passages carry `section_source=content`,
the app flags them in the block note, and `kb_manifest.json` records the split. A heading-
derived label is stronger evidence; if a class leans heavily on rescues, adding the
heading to its PDF is the better fix.

**Severity is not validated.** The lesion mask separates healthy from diseased at the
median but the distributions overlap in the tails. Treat the percentage as indicative
until it is checked against hand-scored leaves.

**Classification metrics may be optimistic.** If notebook 01's train/val/test split
was carved at random from augmented near-duplicates, the grape and tomato F1 scores
are inflated. Run the duplication audit before quoting them.
