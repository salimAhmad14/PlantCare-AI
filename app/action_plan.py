"""Turns a finished report into an ordered action plan: "what do I do now".

This is EXTRACTIVE, deliberately. Every line comes from a retrieved passage and
keeps the citation it arrived with. Nothing here writes new agronomy.

That constraint is the whole reason the rest of the pipeline is trustworthy. A
summariser that paraphrased "apply myclobutanil at first sign of infection" into
its own words would be inventing a dosage recommendation with a citation attached
to it, which is worse than having no summary at all — the citation would make the
invention look verified.

So the pipeline is: split passages into sentences, keep the ones that instruct,
rank them, and group them by when they need doing. The ranking is a heuristic; the
sentences are verbatim.
"""

import re

# Verbs that mark a sentence as an instruction rather than a description.
# Weighted: a sentence that opens with one is far more likely to be a step than
# one that merely mentions it in passing.
ACTION_VERBS = {
    "apply", "spray", "remove", "destroy", "burn", "prune", "rogue", "treat",
    "use", "avoid", "monitor", "rotate", "drench", "dust", "collect", "disinfect",
    "sanitise", "sanitize", "plant", "sow", "irrigate", "drain", "mulch", "stake",
    "thin", "cover", "seal", "replace", "cut", "dispose", "select", "maintain",
    "ensure", "provide", "increase", "reduce", "space", "water", "harvest",
    "scout", "inspect", "check", "follow", "adopt", "practise", "practice",
    "deep", "grow", "keep", "start", "begin", "stop", "protect", "control",
}

# Concrete specifics raise a sentence's priority: a named chemical or a rate is
# more actionable than a general principle.
SPECIFIC = re.compile(
    r"\b(myclobutanil|mancozeb|captan|tebuconazole|azoxystrobin|propiconazole|"
    r"chlorothalonil|copper|sulphur|sulfur|bordeaux|trichoderma|pseudomonas|"
    r"bacillus|carbendazim|difenoconazole|hexaconazole|neem|streptomycin|"
    r"metalaxyl|thiophanate|carboxin|imidacloprid)\w*\b"
    r"|\b\d+(\.\d+)?\s?(%|g|kg|ml|l|litre|liter|ppm|days?|weeks?)\b", re.I)

# "Vine Surgery (Trunk Renewal): cut off the trunk below..." - these documents
# label their points. The label makes a good step title.
LABELLED = re.compile(r"^([A-Z][A-Za-z0-9 &()/'-]{2,48}):\s+(.*)$")

# Sentences that describe rather than instruct.
NON_ACTION = re.compile(
    r"^(the|a|an|this|these|those|it|there|infection|symptoms?|spores?|"
    r"disease|fungus|damage|yield|losses)\b", re.I)


def _sentences(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def _score(sentence):
    """How much does this sentence read like an instruction?"""
    s = sentence.strip()
    body = s
    label = None

    m = LABELLED.match(s)
    if m:
        label, body = m.group(1), m.group(2)

    words = re.findall(r"[a-z]+", body.lower())
    if not words:
        return 0, None, s

    score = 0
    if words[0] in ACTION_VERBS:
        score += 5                       # imperative opening
    hits = len(ACTION_VERBS & set(words[:14]))
    score += 2 * hits
    if label:
        score += 2                       # the document itself flagged it as a point
    if SPECIFIC.search(s):
        score += 3                       # names a chemical or a rate
    if NON_ACTION.match(body):
        score -= 2
    if len(s) > 320:
        score -= 1                       # long sentences are usually exposition

    # Return the body without the label - the caller renders the label separately,
    # and keeping it in both places prints "Pruning: Pruning: remove...".
    return score, label, (body if label else s)


def _steps_from(passages, limit, seen):
    """Rank the instructive sentences in these passages and take the top `limit`."""
    scored = []
    for p in passages:
        for sentence in _sentences(p.get("text", "")):
            key = sentence[:70].lower()
            if key in seen:
                continue
            score, label, text = _score(sentence)
            if score >= 4:
                scored.append((score, label, text, p.get("citation", "")))

    scored.sort(key=lambda x: -x[0])
    out = []
    for score, label, text, citation in scored:
        key = text[:70].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"label": label, "text": text, "citation": citation})
        if len(out) >= limit:
            break
    return out


URGENCY = {
    "High":     ("Act today", "Severe infection. Start control now and get an "
                              "agriculture officer to look at the field."),
    "Moderate": ("Act this week", "Established infection. Begin control and "
                                  "re-check the crop in 5-7 days."),
    "Mild":     ("Watch closely", "Early stage. Remove affected leaves and inspect "
                                  "every 3-4 days before spraying anything."),
    "Healthy":  ("No action needed", "Keep up routine monitoring."),
    "Unavailable": ("Confirm by eye", "Leaf area could not be measured from this "
                                      "photo, so the urgency below is based on the "
                                      "diagnosis alone."),
}


def build(report):
    """-> {'urgency', 'urgency_note', 'groups': [{'when', 'steps'}], 'closing'}"""
    level = report.get("severity_level", "Unavailable")
    urgency, note = URGENCY.get(level, URGENCY["Unavailable"])

    plan = {"urgency": urgency, "urgency_note": note, "groups": [], "closing": []}

    if report.get("is_healthy"):
        plan["urgency"], plan["urgency_note"] = URGENCY["Healthy"]
        plan["groups"].append({"when": "Keep doing", "steps": [
            {"label": None, "citation": "",
             "text": "No disease was detected. Continue your normal watering, spacing "
                     "and field sanitation, and photograph any leaf that starts to "
                     "look different."}]})
        return plan

    if report.get("low_confidence"):
        gate = report.get("confidence_gate", 0.7)
        plan["urgency"] = "Get the diagnosis confirmed first"
        plan["urgency_note"] = (
            f"Confidence is {report.get('confidence_pct')}, below the "
            f"{gate * 100:.0f}% threshold. No treatment steps are shown, because "
            "acting on the wrong diagnosis wastes money and can make things worse.")
        plan["groups"].append({"when": "Do this first", "steps": [
            {"label": None, "citation": "",
             "text": "Take two or three fresh photographs in daylight, against a plain "
                     "background, showing both the top and underside of an affected "
                     "leaf, and run them through again."},
            {"label": None, "citation": "",
             "text": "If the result is still uncertain, show the leaves to your nearest "
                     "Krishi Vigyan Kendra (KVK) or block agriculture officer before "
                     "buying any chemical."}]})
        return plan

    sections = report.get("sections", {})
    seen = set()

    now = _steps_from(sections.get("precaution", {}).get("passages", []), 4, seen)
    if now:
        plan["groups"].append({"when": "Do now", "steps": now})

    # Longer-term steps come from the KB's `prevention` section, fetched by report.py
    # for this purpose. The symptom/cause/transmission blocks are deliberately NOT
    # used: they explain what is happening, not what to do, and mining them for
    # instructions produces lines like "Wind blows spores into open flowers" dressed
    # up as a step.
    later = _steps_from(report.get("plan_sources", []), 3, seen)
    if later:
        plan["groups"].append({"when": "Keep doing this season", "steps": later})

    if not plan["groups"]:
        plan["groups"].append({"when": "Do now", "steps": [
            {"label": None, "citation": "",
             "text": "The knowledge base has no step-by-step control text for this "
                     "disease. Take the diagnosis and these photographs to your nearest "
                     "KVK and ask for a treatment schedule."}]})

    plan["closing"].append(
        "Read the full report before buying anything. Every step above is quoted from "
        "the source document shown beside it.")
    if report.get("tier") != "A":
        plan["closing"].append(
            "Some of this guidance is general crop advice rather than disease-specific. "
            "Confirm it with an agriculture officer first.")
    if report.get("severity_conflict"):
        plan["closing"].append(
            "The lesion measurement and the diagnosis disagreed on this photo. Judge "
            "the urgency by looking at the plant, not by the percentage.")
    return plan


def to_lines(plan):
    """Flat text rendering, used by the PDF, the DOCX and the .txt export."""
    lines = [plan["urgency"].upper(), plan["urgency_note"], ""]
    for group in plan["groups"]:
        lines += [group["when"].upper(), "-" * len(group["when"])]
        for i, step in enumerate(group["steps"], 1):
            head = f"{i}. "
            if step["label"]:
                head += f"{step['label']}: "
            lines.append(head + step["text"])
            if step["citation"]:
                lines.append(f"     [{step['citation']}]")
        lines.append("")
    for c in plan["closing"]:
        lines.append("! " + c)
    return lines
