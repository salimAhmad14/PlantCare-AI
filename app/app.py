"""PlantCare Flask app.

Routes
    GET  /               upload form
    POST /analyse        run the pipeline, render the report page
    POST /api/analyse    same pipeline, JSON response (for an external frontend)
    POST /download       plain-text report download
    GET  /health         readiness check - tells you which artifacts are missing

Run:
    python app.py                       # http://127.0.0.1:5000
    PLANTCARE_DEBUG=1 python app.py     # auto-reload off; debug pages on

Production:
    set PLANTCARE_SECRET_KEY, leave PLANTCARE_DEBUG unset, and serve with
    waitress (Windows) or gunicorn (Linux):
        waitress-serve --port=5000 app:app
        gunicorn -w 2 -b 0.0.0.0:5000 app:app
"""

import io
import os
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import config
import export
import leaf_analysis
import advisor
import differential
import report as report_mod

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_MB * 1024 * 1024

if not config.SECRET_KEY:
    raise RuntimeError(
        "PLANTCARE_SECRET_KEY is not set and debug mode is off. Set the environment "
        "variable, or run with PLANTCARE_DEBUG=1 for local development.")
app.secret_key = config.SECRET_KEY


@app.after_request
def add_cors(resp):
    # Only emits CORS headers when an origin is explicitly configured. The previous
    # version hardcoded "*", which let any site on the internet post to /api/analyse.
    if config.CORS_ORIGIN:
        resp.headers["Access-Control-Allow-Origin"] = config.CORS_ORIGIN
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp


def _check_upload(files):
    if "image" not in files or files["image"].filename == "":
        return None, "Please choose a leaf photograph first."
    f = files["image"]
    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        return None, f"Unsupported file type '{ext}'. Use JPG, PNG, BMP or WEBP."
    return f, None


# Finished reports, kept so the PDF / Word buttons do not have to re-upload the
# image and re-run the CNN. Bounded, in-process, deliberately not a database: a
# diagnosis is worth keeping for the minute it takes to click Download, not longer.
#
# This assumes ONE gunicorn worker, which is what the Dockerfile runs (Milvus Lite
# is a single-writer database). With several workers a download could land on a
# worker that never saw the report; move this to Redis if you ever scale out.
_REPORTS = OrderedDict()
_REPORT_LIMIT = 32


def _remember(report):
    rid = uuid.uuid4().hex[:16]
    _REPORTS[rid] = report
    while len(_REPORTS) > _REPORT_LIMIT:
        _REPORTS.popitem(last=False)
    return rid


def _recall(rid):
    report = _REPORTS.get(rid)
    if report is not None:
        _REPORTS.move_to_end(rid)
    return report


@app.route("/")
def index():
    return render_template("index.html", max_mb=config.MAX_CONTENT_MB)


@app.route("/analyse", methods=["POST"])
def analyse():
    f, err = _check_upload(request.files)
    if err:
        return render_template("index.html", error=err,
                               max_mb=config.MAX_CONTENT_MB), 400
    try:
        bgr = leaf_analysis.read_upload(f)
        rep = report_mod.build(bgr, filename=secure_filename(f.filename))
    except FileNotFoundError as exc:
        return render_template("index.html", error=str(exc),
                               max_mb=config.MAX_CONTENT_MB), 500
    except Exception as exc:                        # noqa: BLE001
        traceback.print_exc()
        return render_template("index.html", error=f"Analysis failed: {exc}",
                               max_mb=config.MAX_CONTENT_MB), 500
    return render_template("result.html", r=rep, report_id=_remember(rep))


@app.route("/api/analyse", methods=["POST", "OPTIONS"])
def api_analyse():
    if request.method == "OPTIONS":
        return "", 204
    f, err = _check_upload(request.files)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    try:
        bgr = leaf_analysis.read_upload(f)
        rep = report_mod.build(bgr, filename=secure_filename(f.filename))
    except Exception as exc:                        # noqa: BLE001
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(exc)}), 500

    payload = report_mod.to_json(rep)
    if request.args.get("images") == "1":
        payload["images"] = rep["images"]            # data: URIs, large
    return jsonify({"ok": True, "report": payload})


@app.route("/report/<rid>.<ext>")
def download(rid, ext):
    """PDF / Word / plain-text export of a report already produced this session."""
    rep = _recall(rid)
    if rep is None:
        # The cache is bounded, so an old link can legitimately expire.
        return render_template(
            "index.html", max_mb=config.MAX_CONTENT_MB,
            error="That report is no longer held in memory. Upload the leaf again "
                  "to regenerate it."), 404

    try:
        if ext == "pdf":
            buf, mime = export.to_pdf(rep), "application/pdf"
        elif ext == "docx":
            buf, mime = (export.to_docx(rep),
                         "application/vnd.openxmlformats-officedocument"
                         ".wordprocessingml.document")
        elif ext == "txt":
            buf = io.BytesIO(report_mod.to_text(rep).encode("utf-8"))
            mime = "text/plain"
        else:
            return Response("Unsupported format", status=404, mimetype="text/plain")
    except Exception as exc:                            # noqa: BLE001
        traceback.print_exc()
        return Response(f"Export failed: {exc}", status=500, mimetype="text/plain")

    return send_file(buf, mimetype=mime, as_attachment=True,
                     download_name=export.filename(rep, ext))


@app.route("/health")
def health():
    kb_numpy = config.KB_CHUNKS.exists() and config.KB_VECTORS.exists()
    checks = {
        "cnn_weights": config.CNN_WEIGHTS.exists(),
        "class_index": config.CLASS_INDEX.exists(),
        "severity_module": config.SEVERITY_MODULE.exists(),
        "kb_manifest": config.KB_MANIFEST.exists(),
        "knowledge_base": kb_numpy or config.KB_DB.exists(),
        "run_config": config.RUN_CONFIG_PATH.exists(),
        "ood_gaussian": config.OOD_GAUSSIAN.exists(),
        "preprocessing_module": config.PREPROCESS_MODULE.exists(),
    }
    missing = [k for k, v in checks.items() if not v]

    body = {"ok": not missing, "artifacts": checks, "missing": missing,
            "artifact_dir": str(config.ARTIFACT_DIR)}
    if not missing:
        try:
            import classifier
            import knowledge_base as kb
            body["knowledge_base_info"] = kb.health()
            body["classes"] = len(classifier.class_names())
            body["confidence_gate"] = classifier.confidence_gate()
            body["warnings"] = classifier.warnings()
            body["screens"] = {
                "confidence_gate": config.CONFIDENCE_GATE,
                "ood_threshold": config.OOD_THRESHOLD,
                "vegetation_range": [config.VEGETATION_MIN, config.VEGETATION_MAX],
                "preprocessing_applied": config.APPLY_PREPROCESSING,
            }
            body["advisory"] = advisor.status()
            body["differential"] = differential.health()

            # Fingerprint check. The commonest silent failure is copying half a rebuild:
            # a class_index from one run and a KB from another. Both are reported so a
            # mismatch is visible on /health rather than in a wrong answer.
            kb_classes = set(kb.manifest().get("classes", []))
            model_classes = set(classifier.class_names())
            if kb_classes and model_classes and kb_classes != model_classes:
                body["ok"] = False
                body["fingerprint_error"] = {
                    "message": ("class_index.json and kb_manifest.json describe "
                                "different class sets - the artifacts come from "
                                "different builds."),
                    "in_model_only": sorted(model_classes - kb_classes),
                    "in_kb_only": sorted(kb_classes - model_classes),
                }
        except Exception as exc:                     # noqa: BLE001
            body["ok"] = False
            body["load_error"] = f"{type(exc).__name__}: {exc}"
            return jsonify(body), 503
    return jsonify(body), (200 if body["ok"] else 503)


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html",
                           error=f"That file is larger than {config.MAX_CONTENT_MB} MB.",
                           max_mb=config.MAX_CONTENT_MB), 413


def _startup_check():
    kb_ok = ((config.KB_CHUNKS.exists() and config.KB_VECTORS.exists())
             or config.KB_DB.exists())
    missing = [n for n, ok in [
        ("best_cnn.pt", config.CNN_WEIGHTS.exists()),
        ("class_index.json", config.CLASS_INDEX.exists()),
        ("severity.py", config.SEVERITY_MODULE.exists()),
        ("chunks.csv + embeddings.npy (or plantcare.db)", kb_ok)] if not ok]
    if missing:
        print("\n  MISSING ARTIFACTS:", ", ".join(missing))
        print(f"  Put them in: {config.ARTIFACT_DIR}")
        print("  The app will still start; /health lists what is missing.\n")


if __name__ == "__main__":
    _startup_check()
    # use_reloader=False: the reloader starts a second process, and both would open
    # the Milvus Lite database - the second one fails on the lock file.
    app.run(host=os.environ.get("PLANTCARE_HOST", "127.0.0.1"),
            port=int(os.environ.get("PLANTCARE_PORT", "5000")),
            debug=config.DEBUG,
            use_reloader=False)
