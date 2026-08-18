"""Central configuration.

Everything that differs between your laptop and a server is read from an
environment variable with a sensible local default, so nothing has to be edited
in code to deploy.

    PLANTCARE_SECRET_KEY    Flask session key. REQUIRED when DEBUG is off.
    PLANTCARE_CORS_ORIGIN   Allowed origin for /api/analyse. Default: none.
    PLANTCARE_MILVUS_URI    Milvus server URI. Set it to use a server instead of
                            the embedded / numpy backend.
    PLANTCARE_KB_BACKEND    Force a backend: "numpy" | "milvus" | "auto" (default).
    PLANTCARE_DEBUG         "1" to enable Flask debug. Default: off.
    PLANTCARE_ARTIFACT_DIR  Where the model + KB artifacts live. Default: ./artifacts
"""

import os
from pathlib import Path

# .env is loaded HERE, not in app.py. config is the first module everything imports, so
# loading it anywhere else means whichever module gets imported first wins - and if that
# is knowledge_base rather than app, config reads the environment before .env is applied
# and the secret key looks missing.
#
# .resolve() matters: run as `python app.py`, __file__ is the relative string "app.py",
# so .parent and .parent.parent are both "." and the project-root .env is never found.
try:
    from dotenv import load_dotenv
    _HERE = Path(__file__).resolve().parent
    for _cand in (_HERE / ".env", _HERE.parent / ".env"):
        if _cand.exists():
            load_dotenv(_cand)
            print(f"[env] loaded {_cand}")
            break
    else:
        print("[env] no .env found - reading os.environ only")
except ImportError:
    print("[env] python-dotenv not installed - reading os.environ only")

# --- paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"

# Optional override, for pointing at a freshly built KB without moving files. A path
# that does not exist falls back with a warning instead of taking the app down - an
# environment variable set to a stale or mistyped path should not look like a missing
# knowledge base.
_override = os.environ.get("PLANTCARE_ARTIFACT_DIR", "").strip().strip('"').strip("'")
if _override and not Path(_override).is_dir():
    print(f"[config] PLANTCARE_ARTIFACT_DIR={_override!r} is not a directory - "
          f"using {DEFAULT_ARTIFACT_DIR} instead.\n"
          f"[config] Clear it with:  Remove-Item Env:\\PLANTCARE_ARTIFACT_DIR")
    _override = ""
ARTIFACT_DIR = Path(_override) if _override else DEFAULT_ARTIFACT_DIR
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

RUN_CONFIG_PATH = ARTIFACT_DIR / "run_config.json"
OOD_GAUSSIAN = ARTIFACT_DIR / "ood_gaussian.npz"
PREPROCESS_MODULE = ARTIFACT_DIR / "preprocessing.py"

# Notebook 01 exports every gate and threshold it was calibrated with. Reading them
# here means the app cannot drift from the notebook: change a gate there, re-copy the
# file, and the app follows. Hardcoded defaults are only a floor for a missing file.
import json as _json
try:
    RUN_CONFIG = _json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
except Exception:                                              # noqa: BLE001
    RUN_CONFIG = {}
_REJECT = RUN_CONFIG.get("rejection", {})

CNN_WEIGHTS = ARTIFACT_DIR / "best_cnn.pt"
CLASS_INDEX = ARTIFACT_DIR / "class_index.json"
SEVERITY_MODULE = ARTIFACT_DIR / "severity.py"

KB_MANIFEST = ARTIFACT_DIR / "kb_manifest.json"
KNOWLEDGE_GRAPH = ARTIFACT_DIR / "knowledge_graph.json"
KB_DB = ARTIFACT_DIR / "plantcare.db"          # Milvus Lite (Linux/macOS only)
KB_CHUNKS = ARTIFACT_DIR / "chunks.csv"        # numpy backend
KB_VECTORS = ARTIFACT_DIR / "embeddings.npy"   # numpy backend

# --- web -----------------------------------------------------------------
DEBUG = os.environ.get("PLANTCARE_DEBUG", "") == "1"
SECRET_KEY = os.environ.get("PLANTCARE_SECRET_KEY") or ("dev-only-key" if DEBUG else None)
CORS_ORIGIN = os.environ.get("PLANTCARE_CORS_ORIGIN", "")   # empty = no CORS headers
MAX_CONTENT_MB = 16
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# --- retrieval -----------------------------------------------------------
MILVUS_URI = os.environ.get("PLANTCARE_MILVUS_URI", "")
# Default is "numpy", not "auto", and that is deliberate.
#
# Milvus Lite has no server-side BM25 Function, so its hybrid probe fails and it falls
# back to DENSE-ONLY retrieval. The numpy backend computes BM25 itself and fuses it with
# the dense scores via RRF, so on this corpus it is strictly the better retriever - and
# at 172 chunks the speed difference is irrelevant. Milvus becomes the right choice when
# you point PLANTCARE_MILVUS_URI at a real server, which does support BM25.
#
# Set PLANTCARE_KB_BACKEND=auto to restore the previous preference order.
KB_BACKEND = os.environ.get("PLANTCARE_KB_BACKEND", "numpy").lower()
RRF_K = 60          # reciprocal-rank-fusion constant
BM25_K1 = 1.5
BM25_B = 0.75

# --- inference -----------------------------------------------------------
# All four screens come from notebook 01's run_config.json where available.
CONFIDENCE_GATE = float(_REJECT.get("confidence_gate",
                                    RUN_CONFIG.get("confidence_gate", 0.80)))
OOD_THRESHOLD = _REJECT.get("ood_threshold")          # None -> Mahalanobis screen off
VEGETATION_MIN = float(_REJECT.get("vegetation_min", 0.06))
VEGETATION_MAX = float(_REJECT.get("vegetation_max", 0.97))
REJECT_MESSAGES = dict({
    "no_leaf": "Cannot classify - no leaf detected in the image",
    "unknown_leaf": "Cannot classify - this leaf type is not in the training dataset",
    "low_confidence": "Cannot classify - image is unclear or leaf type not recognized",
}, **_REJECT.get("messages", {}))
REJECT_MESSAGES.setdefault(
    "frame_filled",
    "Cannot measure this image - the leaf fills the entire frame")

# Notebook 01 applies preprocess_image() during training AND at inference, so the app
# must too. Skipping it here is a train/serve mismatch: the model would see a different
# input distribution from the one it was validated on.
APPLY_PREPROCESSING = os.environ.get("PLANTCARE_PREPROCESS", "1") != "0"

TOP_K = 3
IMG_SIZE = int(RUN_CONFIG.get("img_size", 224))
MEAN = RUN_CONFIG.get("mean", [0.485, 0.456, 0.406])
STD = RUN_CONFIG.get("std", [0.229, 0.224, 0.225])

# Report block -> (KB section_type, query template)
#
# These MUST match the section vocabulary written by notebook 02. The rewritten
# notebook emits seven section types (identity/overview/pathogen/transmission/
# symptom/prevention/treatment) instead of four, which is what lets "cause" and
# "reason" draw on genuinely different pools. The old mapping pointed both at
# "biology", so cross-block dedup always emptied one of them.
#
# kb_manifest.json carries the mapping the KB was built with; /health surfaces it.
SECTION_QUERIES = {
    "symptoms":   ("symptom",      "symptoms, appearance and identification of {name} on {crop} leaves"),
    "cause":      ("pathogen",     "causal organism and pathogen responsible for {name} in {crop}"),
    "reason":     ("transmission", "how {name} spreads, transmission and favourable conditions"),
    "precaution": ("treatment",    "treatment, chemical and biological control of {name}"),
    "prevention": ("prevention",   "prevention, cultural practices and sanitation for {name}"),
}

# Legacy four-type KBs (section_type in {general, symptom, biology, management}).
# Selected automatically when kb_manifest.json reports the old vocabulary.
SECTION_QUERIES_LEGACY = {
    "symptoms":   ("symptom",    "symptoms, appearance and identification of {name} on {crop} leaves"),
    "cause":      ("biology",    "causal organism and pathogen responsible for {name} in {crop}"),
    "reason":     ("biology",    "favourable weather conditions, humidity, temperature and spread of {name}"),
    "precaution": ("management", "control measures, fungicide spray, dosage and prevention of {name}"),
    "prevention": ("management", "cultural practices, sanitation and prevention of {name}"),
}

SECTION_HEADINGS = {
    "symptoms":   "Symptoms",
    "cause":      "Cause",
    "reason":     "Why it spread",
    "precaution": "Treatment",
    "prevention": "Prevention and cultural practice",
}

# Order the report renders them in.
SECTION_ORDER = ["symptoms", "cause", "reason", "precaution", "prevention"]

# --- presentation --------------------------------------------------------
# The lesion mask is chroma/darkness based: it measures brown necrotic spots well and
# DIFFUSE symptoms badly - leaf mold's pale blotches, TYLCV chlorosis, mite stippling.
# A correctly classified leaf-mold image measured 0.1% and landed in "Healthy". Printing
# "no action needed" there would tell a farmer to ignore a confirmed disease, so a
# diseased leaf whose band is Healthy or Unavailable gets its severity treated as
# UNMEASURED instead of as good news.
#
# Principle: the classifier decides WHETHER there is disease; the mask only decides
# HOW MUCH.
SEVERITY_ADVICE_DISEASED = {
    "Healthy": ("A disease was identified but the lesion mask measured almost no "
                "affected area. Treat the percentage as unmeasured, not as an "
                "all-clear, and inspect the leaf yourself within the week."),
    "Unavailable": ("A disease was identified but the leaf area could not be measured "
                    "from this image. Inspect the leaf yourself within the week."),
    "Mild": "Early stage. Remove affected leaves and monitor every 3-4 days.",
    "Moderate": ("Established infection. Begin the control measures below and re-check "
                 "in 5-7 days."),
    "High": ("Severe. Act immediately and consult your local KVK or agriculture "
             "officer."),
}

SEVERITY_URGENCY = {
    "Healthy": "none - routine monitoring",
    "Mild": "low - act within the week",
    "Moderate": "moderate - act within 2-3 days",
    "High": "high - act immediately",
    "Unavailable": "unknown - severity could not be measured",
}

SEVERITY_COLOUR = {
    "Healthy": "#2e7d32",
    "Mild": "#fbc02d",
    "Moderate": "#ef6c00",
    "High": "#c62828",
    "Unavailable": "#546e7a",
}

# Keyed by band for a leaf the classifier called HEALTHY. A diseased leaf never reads
# from this dict directly - see SEVERITY_ADVICE_DISEASED and report.resolve_urgency.
SEVERITY_ADVICE = {
    "Healthy": "No action needed. Continue routine monitoring.",
    "Mild": ("Early stage. Remove affected leaves and monitor every 3-4 days"
             " before treating."),
    "Moderate": ("Established infection. Begin the control measures below and"
                 " re-check in 5-7 days."),
    "High": ("Severe. Act immediately and consult your local KVK or agriculture"
             " officer."),
    "Unavailable": ("Leaf area could not be measured reliably from this image -"
                    " severity not reported."),
}


# --- LLM advisory --------------------------------------------------------
# Optional. With no key the app still produces the full extractive report; the prose
# section is simply absent. Nothing in the pipeline depends on the model being up.
#
# The key is read from the environment (python-dotenv loads .env at startup), never
# from a file the app serves.
LLM_ENABLED = os.environ.get("PLANTCARE_LLM", "1") != "0"
LLM_PROVIDER = os.environ.get("PLANTCARE_LLM_PROVIDER", "groq").lower()

_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        # Free-tier catalogues change without notice, so advisor.py discovers what is
        # live and takes the first of these that the API actually serves.
        "prefer": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b",
                   "qwen3-32b", "llama-3.1-8b-instant"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "prefer": ["meta-llama/llama-3.3-70b-instruct:free",
                   "qwen/qwen-2.5-72b-instruct:free"],
    },
}
_P = _PROVIDERS.get(LLM_PROVIDER, _PROVIDERS["groq"])
LLM_BASE_URL = os.environ.get("PLANTCARE_LLM_BASE_URL", _P["base_url"])
LLM_KEY_ENV = _P["key_env"]
LLM_PREFER = _P["prefer"]
LLM_TEMPERATURE = 0.15          # low: rewrite the sources, do not be creative
LLM_TIMEOUT = 90
LLM_MIN_INTERVAL = 1.0          # seconds between calls; free tiers cap tokens/minute
LLM_MAX_RETRIES = 3

# --- differential --------------------------------------------------------
# Look-alike diseases are DERIVED from per-class symptom-embedding centroids, not
# hand-written: whichever classes in the same crop are described with the most similar
# symptom language. On this KB that recovers early blight <-> target spot (both
# "concentric rings"), septoria <-> bacterial spot, and early <-> late blight.
DIFFERENTIAL_ENABLED = os.environ.get("PLANTCARE_DIFFERENTIAL", "1") != "0"
DIFFERENTIAL_K = 2              # look-alike diseases to contrast against
DIFFERENTIAL_PASSAGES = 1       # symptom passages per look-alike
