"""LLM advisory layer.

Turns the retrieved passages into readable prose via an OpenAI-compatible API
(Groq by default), then AUDITS the result and throws it away if it fails.

Design, in one line: the model rewrites evidence it is given, and anything it adds
that is not in that evidence is deleted rather than published.

Three fences:

1. Closed book. Only the numbered sources; if they do not say it, say so.
2. Mandatory [S#] citations on every sentence.
3. A post-generation audit (audit_answer) that checks every number-with-a-unit,
   every chemical-looking token and every citation index against the source text -
   plus a check that the model has not claimed to look at the photograph.

The third is the one that actually catches things. If it fails, report.py keeps the
extractive sections it already built and shows no prose. The app is fully functional
with no API key at all.
"""

import json
import os
import re
import time

import requests

import config

# --------------------------------------------------------------- provider
_STATE = {"model": None, "checked": False, "error": None}


def _key():
    return (os.environ.get(config.LLM_KEY_ENV, "") or "").strip()


def available():
    """True when a key is present and a usable model was discovered."""
    if _STATE["checked"]:
        return _STATE["model"] is not None
    _STATE["checked"] = True

    if not _key():
        _STATE["error"] = f"{config.LLM_KEY_ENV} not set"
        return False
    try:
        r = requests.get(f"{config.LLM_BASE_URL}/models", timeout=20,
                         headers={"Authorization": f"Bearer {_key()}"})
        r.raise_for_status()
        live = [m["id"] for m in r.json().get("data", [])]
    except Exception as exc:                                   # noqa: BLE001
        _STATE["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return False

    # The free-tier catalogue changes without notice, so the served model is
    # discovered rather than hardcoded.
    chosen = next((m for m in config.LLM_PREFER if m in live), None)
    if chosen is None:
        chosen = next((m for m in live
                       if not re.search(r"whisper|guard|tts|embed|rerank|vision", m, re.I)),
                      None)
    _STATE["model"] = chosen
    if chosen is None:
        _STATE["error"] = "no usable chat model in the provider catalogue"
    return chosen is not None


def status():
    ok = available()
    return {"enabled": ok, "model": _STATE["model"], "provider": config.LLM_PROVIDER,
            "base_url": config.LLM_BASE_URL, "error": _STATE["error"]}


_last_call = [0.0]


def _chat(system, user):
    for attempt in range(config.LLM_MAX_RETRIES):
        wait = config.LLM_MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()

        r = requests.post(
            f"{config.LLM_BASE_URL}/chat/completions", timeout=config.LLM_TIMEOUT,
            headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
            json={"model": _STATE["model"], "temperature": config.LLM_TEMPERATURE,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]})
        if r.status_code == 429:
            back = float(r.headers.get("Retry-After", config.LLM_MIN_INTERVAL * (attempt + 2)))
            time.sleep(min(back, 30.0))
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError("rate limited after retries")


# --------------------------------------------------------------- the prompt
SYSTEM = """You are an agricultural extension advisor writing a field advisory.

You are one stage in a pipeline. An image classifier has already identified the disease
and a segmentation model has already measured how much leaf area is affected. You did NOT
see the photograph and you cannot see it now.

ABSOLUTE RULES

1. NEVER describe the photograph. You have not seen it. Do not write "the image shows",
   "the leaf in the photo", "visible in this sample", or anything similar. Write about
   what the disease TYPICALLY does, citing a source: "Early blight typically begins on
   the oldest lowest leaves [S3]."

2. Use ONLY the numbered SOURCES. No outside knowledge. If the sources do not cover
   something, write exactly: "The sources do not state this."

3. NEVER invent a chemical name, a dose, a rate, a concentration, a spray interval or a
   temperature. If a number is not in the sources, it does not go in your answer.

4. Every sentence ends with one or more citation markers: [S1], or [S2][S5].

5. The DIAGNOSIS and the SEVERITY BAND are given to you as facts. Do not overturn them,
   re-diagnose, or dispute the percentage. Your job is to explain and act on them.

6. Where you contrast the diagnosis against a look-alike disease, ground every contrast
   in a cited source passage from the sources given.

Write in plain language a farmer can act on. Short sentences. No preamble, no sign-off."""


TEMPLATE = """CASE

  Crop                  : {crop}
  Diagnosis             : {disease}
  Classifier confidence : {confidence}
  Leaf area affected    : {severity}  -> band: {band}
  Urgency (fixed by the pipeline): {urgency}

SOURCES FOR {disease_upper}

{sources}
{differential_block}{caveat_block}
TASK

Write the advisory using exactly these headings, in this order:

## What this disease is
One or two sentences: the pathogen and what kind of organism it is. Cite.

## Why this diagnosis fits
Describe the symptoms this disease TYPICALLY produces, from the sources. Do not claim to
have observed them. Cite every sentence.
{differential_task}
## How it spreads
Cite. Two sentences maximum.

## What to do now
Actions justified by the "{urgency}" urgency level, drawn from the treatment and
prevention sources. Every action cited. If the sources give no treatment, say so - do not
substitute a general recommendation.

## What to keep doing
Preventive and cultural practices for the rest of the season. Cite.

## Limits of this advice
Two sentences. State plainly that the diagnosis came from an image classifier and not a
laboratory test, and that chemical choices must follow local registration and label
rates. No citation needed for this heading only.
"""

DIFFERENTIAL_TASK = """
## What else it could look like
For each look-alike disease listed above, give ONE sentence naming a cited symptom
difference that separates it from {disease}. Format each as:
  - **<Other disease>:** <the difference> [S#]
If the sources give you no basis to separate two diseases, say so for that disease
instead of inventing a distinction.
"""


def build_prompt(case, blocks, differential, caveats):
    """blocks: {block_key: [passage, ...]}; differential: {other_label: [passage, ...]}"""
    sources, meta = [], []
    for key, passages in blocks.items():
        for p in passages:
            meta.append(p)
            sources.append(f"[S{len(meta)}] ({key} | {p['citation']})\n      {p['text']}")

    diff_lines = []
    for other, passages in (differential or {}).items():
        diff_lines.append(f"  --- {other} ---")
        for p in passages:
            meta.append(p)
            diff_lines.append(f"[S{len(meta)}] ({p['citation']})\n      {p['text']}")

    diff_block = ""
    diff_task = ""
    if diff_lines:
        diff_block = ("\nSYMPTOM PASSAGES FOR LOOK-ALIKE DISEASES\n"
                      "(the diseases in this crop whose described symptoms are closest to the\n"
                      "diagnosis; given so you can contrast, NOT so you can re-diagnose)\n\n"
                      + "\n\n".join(diff_lines) + "\n")
        diff_task = DIFFERENTIAL_TASK.format(disease=case["disease"])

    cav = ("\nIMPORTANT CONTEXT\n\n  " + "\n  ".join(caveats) + "\n") if caveats else ""

    user = TEMPLATE.format(
        crop=case["crop"], disease=case["disease"],
        disease_upper=case["disease"].upper(),
        confidence=case["confidence"], severity=case["severity"],
        band=case["band"], urgency=case["urgency"],
        sources="\n\n".join(sources),
        differential_block=diff_block, differential_task=diff_task,
        caveat_block=cav)
    return user, meta


# --------------------------------------------------------------- the audit
NUM_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:-\s*\d+(?:[.,]\d+)?\s*)?"
    r"(?:%|g|kg|mg|ml|l|litre|liter|lb|oz|ha|acre|day|days|week|weeks|hour|hours|"
    r"year|years|ppm|mesh|cm|mm|inch|inches|°c|°f)\b", re.I)

CHEM_RE = re.compile(
    r"\b(mancozeb|chlorothalonil|copper|captan|myclobutanil|azoxystrobin|difenoconazole|"
    r"tebuconazole|propiconazole|streptomycin|bordeaux|sulfur|sulphur|abamectin|spinosad|"
    r"imidacloprid|spiromesifen|neem|bacillus|trichoderma|maneb|ziram|fosetyl|mefenoxam|"
    r"metalaxyl|acibenzolar|actigard|phytoseiulus|thiophanate|pyraclostrobin|boscalid|"
    r"cymoxanil|famoxadone)\b", re.I)

# The failure that matters most here: the model has not seen the photograph, so a
# sentence describing it is invented - and it reads as the most authoritative line in
# the whole report.
IMAGE_CLAIM_RE = re.compile(
    r"\b(the|this|your)\s+(image|photo|photograph|picture|sample|specimen)\b"
    r"|\bin the (image|photo|picture)\b"
    r"|\b(i|we) can see\b|\bas (seen|shown|visible) (in|on)\b"
    r"|\bthe leaf (shown|pictured|in)\b|\bvisible in\b", re.I)


def _norm(s):
    return re.sub(r"[\s,]+", " ", s.lower()).strip()


def audit_answer(answer, meta, differential=None):
    """-> (ok, [problem, ...]). Any problem means the prose is discarded."""
    source_text = _norm(" ".join(p["text"] for p in meta))
    compact = source_text.replace(" ", "")
    problems = []

    if not re.search(r"\[S\d+\]", answer):
        problems.append("no citation markers at all")

    for m in NUM_UNIT_RE.finditer(answer):
        frag = _norm(m.group(0))
        if frag not in source_text and frag.replace(" ", "") not in compact:
            problems.append(f"unsupported quantity: {m.group(0).strip()!r}")

    for m in CHEM_RE.finditer(answer):
        if m.group(0).lower() not in source_text:
            problems.append(f"unsupported chemical: {m.group(0)!r}")

    for m in re.finditer(r"\[S(\d+)\]", answer):
        if not (1 <= int(m.group(1)) <= len(meta)):
            problems.append(f"citation [S{m.group(1)}] does not exist")

    for m in IMAGE_CLAIM_RE.finditer(answer):
        problems.append(f"claims to have seen the photograph: {m.group(0)!r}")

    if differential:
        sec = re.search(r"##\s*What else it could look like(.+?)(?=\n##|\Z)",
                        answer, re.S | re.I)
        if sec:
            allowed = {d.lower() for d in differential}
            for m in re.finditer(r"^\s*[-*]\s*\*\*(.+?)\*\*", sec.group(1), re.M):
                named = m.group(1).strip().rstrip(":").lower()
                if named and not any(a in named or named in a for a in allowed):
                    problems.append(
                        f"differential names a disease that was never retrieved: "
                        f"{m.group(1)!r}")

    return (not problems), problems


# --------------------------------------------------------------- markdown -> blocks
def _to_sections(markdown):
    """Split the model's '## Heading' output into an ordered list for the template."""
    out, heading, buf = [], None, []
    for line in markdown.splitlines():
        m = re.match(r"^\s*#{2,3}\s*(.+?)\s*$", line)
        if m:
            if heading and buf:
                out.append({"heading": heading, "body": "\n".join(buf).strip()})
            heading, buf = m.group(1), []
        else:
            buf.append(line)
    if heading and buf:
        out.append({"heading": heading, "body": "\n".join(buf).strip()})

    for s in out:
        bullets = [re.sub(r"^\s*[-*]\s*", "", b).strip()
                   for b in s["body"].splitlines() if re.match(r"^\s*[-*]\s", b)]
        s["bullets"] = bullets
        if bullets:
            s["body"] = ""
        s["body"] = re.sub(r"\n{2,}", "\n", s["body"]).strip()
    return [s for s in out if s["body"] or s["bullets"]]


def write_advisory(case, blocks, differential, caveats):
    """-> dict for the template, or None when prose could not be produced safely.

    None is not an error path: it means the report falls back to the extractive
    sections, which are already correct and cited.
    """
    if not config.LLM_ENABLED or not available():
        return None
    if not blocks:
        return None

    user, meta = build_prompt(case, blocks, differential, caveats)
    try:
        answer = _chat(SYSTEM, user)
    except Exception as exc:                                   # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {str(exc)[:160]}",
                "model": _STATE["model"], "sections": [], "sources": []}

    ok, problems = audit_answer(answer, meta, differential=differential)
    if not ok:
        return {"ok": False, "reason": "audit rejected the draft", "problems": problems,
                "model": _STATE["model"], "draft": answer,
                "sections": [], "sources": []}

    return {
        "ok": True, "model": _STATE["model"],
        "sections": _to_sections(answer),
        "sources": [{"n": i + 1, "citation": p["citation"]} for i, p in enumerate(meta)],
        "raw": answer,
    }
