"""Report assembly.

CNN prediction + severity + tiered retrieval -> the blocks shown on the result page:

    diagnosis (name + confidence)   image comparison (original vs lesion overlay)
    severity level                  symptoms
    cause                           reason (why it spread)
    precautions

Generation is EXTRACTIVE. Every sentence in symptoms / cause / reason / precaution
comes from a retrieved chunk and carries its own `Source.pdf p.N` citation. Nothing
is free-generated, because inventing fluent agronomy is exactly the failure mode
that would hurt a farmer acting on it.

Two behaviours worth knowing about:

*   **Reconciliation.** The classifier and the lesion mask are independent. When they
    disagree - a healthy leaf measured at 40% lesion area - one of them is wrong, and
    the old code printed both without comment. Now the contradiction is stated and the
    untrustworthy number is withheld.
*   **Cross-block dedup.** "Cause" and "reason" both draw on `section_type="biology"`,
    which for most diseases is two or three chunks. Without dedup they print the same
    passage twice. With it, "reason" can come back empty - which is the truth about the
    corpus, not a bug.
"""

from datetime import datetime

import action_plan
import advisor
import differential
import classifier
import config
import knowledge_base as kb
import leaf_analysis


def _confidence_band(conf):
    if conf >= 0.90:
        return "Very high"
    if conf >= 0.75:
        return "High"
    if conf >= 0.60:
        return "Moderate"
    return "Low"


def _reconcile(report, analysis, is_healthy):
    """Severity and diagnosis come from independent subsystems. Cross-check them."""
    level = analysis.get("severity_level", "Unavailable")
    pct = analysis.get("severity_percent")

    if is_healthy and level in ("Moderate", "High"):
        report["caveats"].append(
            f"The classifier reports a healthy leaf, but the lesion mask measured "
            f"{pct}% of the leaf area as non-green. These two readings contradict each "
            "other, which almost always means the leaf outline was mis-segmented. The "
            "severity figure is withheld. Re-photograph on a plain background in even "
            "light if the leaf looks damaged to you.")
        report["severity_level"] = "Unavailable"
        report["severity_percent"] = None
        report["severity_display"] = "-"
        report["severity_colour"] = config.SEVERITY_COLOUR["Unavailable"]
        report["severity_advice"] = config.SEVERITY_ADVICE["Unavailable"]
        report["severity_conflict"] = True
        return

    if (not is_healthy) and level in ("Healthy", "Unavailable"):
        # The dangerous direction. The lesion mask is chroma/darkness based, so it
        # measures brown necrosis well and DIFFUSE symptoms badly: a correctly
        # classified leaf-mold image measured 0.1% and landed in "Healthy". Taking the
        # band at face value would print "No action needed" over a confirmed disease.
        report["caveats"].append(
            "The lesion mask found almost no affected area although a disease was "
            "identified. That is consistent with a very early infection, but it can "
            "also mean the symptoms are diffuse - pale mottling, curling or stippling - "
            "which this mask does not measure well. The severity figure is treated as "
            "unmeasured rather than as an all-clear.")
        report["severity_conflict"] = True
        report["severity_advice"] = config.SEVERITY_ADVICE_DISEASED[level]
        report["urgency"] = "low - act within the week (severity not measurable)"


def build(bgr, filename="leaf.jpg"):
    """Full report dict, ready for the Jinja template or JSON serialisation."""
    label, conf, top, extras = classifier.predict_full(bgr)
    crop_title, disease_title, full_title = classifier.pretty(label)
    crop = label.split("_")[0]
    is_healthy = "healthy" in label

    analysis = leaf_analysis.analyse(bgr, crop=crop)
    gate = classifier.confidence_gate()

    # The three refusal screens from notebook 01, in its order. A confidence gate on
    # its own is not enough: softmax over 13 classes always sums to 1, so a photo of a
    # species the model has never seen comes back CONFIDENT and wrong. The leaf-presence
    # and Mahalanobis screens are what catch that, and they must run before retrieval -
    # a refused image should never reach the knowledge base at all.
    reason, detail = classifier.screen(
        analysis.get("leaf_fraction"), conf, extras.get("ood_distance"))
    low_conf = reason == "low_confidence"

    report = {
        "filename": filename,
        "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "label": label,
        "crop": crop_title,
        "disease": disease_title or "Healthy",
        "title": full_title,
        "confidence": conf,
        "confidence_pct": f"{conf * 100:.1f}%",
        "confidence_band": _confidence_band(conf),
        "confidence_gate": gate,
        "low_confidence": low_conf,
        "alternatives": [
            {"label": l, "name": classifier.pretty(l)[2], "pct": f"{p * 100:.1f}%"}
            for l, p in top[1:]
        ],
        "is_healthy": is_healthy,
        "refused": reason is not None,
        "refuse_reason": reason,
        "refuse_detail": detail,
        "refuse_message": config.REJECT_MESSAGES.get(reason, "") if reason else "",
        "ood_distance": extras.get("ood_distance"),
        "ood_threshold": extras.get("ood_threshold"),
        "preprocessed": extras.get("preprocessed"),
        "severity_level": analysis["severity_level"],
        "severity_percent": analysis["severity_percent"],
        "severity_display": analysis["severity_pct_display"],
        "severity_colour": config.SEVERITY_COLOUR.get(analysis["severity_level"], "#546e7a"),
        "severity_advice": (config.SEVERITY_ADVICE if is_healthy
                            else config.SEVERITY_ADVICE_DISEASED).get(
                                analysis["severity_level"], ""),
        "urgency": config.SEVERITY_URGENCY.get(analysis["severity_level"], ""),
        "advisory": None,
        "differential": [],
        "severity_conflict": False,
        "lesion_count": analysis["lesion_count"],
        "largest_lesion_pct": analysis["largest_lesion_pct"],
        "mask_ok": analysis["mask_ok"],
        "images": analysis["images"],
        "sections": {},
        "citations": [],
        "caveats": [],
        "tier": "H" if is_healthy else "C",
        "retrieval": kb.backend.name,
    }

    if not analysis["mask_ok"]:
        report["caveats"].append(
            "The leaf outline could not be measured reliably in this photograph, so the "
            "severity figure is withheld. Retake on a plain background in even light.")

    _reconcile(report, analysis, is_healthy)

    # ---- not a leaf, or not a leaf we know: refuse outright ---------------
    if reason in ("no_leaf", "unknown_leaf", "frame_filled"):
        report["caveats"].insert(0, config.REJECT_MESSAGES[reason] + f" ({detail}).")
        report["disease"] = "Not classified"
        report["title"] = "Not classified"
        report["alternatives"] = []
        report["severity_level"] = "Unavailable"
        report["severity_percent"] = None
        report["severity_display"] = "-"
        report["severity_advice"] = ""
        report["urgency"] = ""
        report["sections"]["summary"] = {
            "heading": "Cannot classify this image",
            "tier": "C",
            "passages": [],
            "note": (config.REJECT_MESSAGES[reason] + ". " + (
                "Photograph a single leaf so it fills most of the frame, on a plain "
                "background in even light."
                if reason == "no_leaf" else
                "Step back so a margin of plain background is visible around the leaf, "
                "then retake the photograph."
                if reason == "frame_filled" else
                "This looks unlike anything in the training data, so no diagnosis is "
                "offered. The model covers grape and tomato leaves only.")),
        }
        report["plan"] = action_plan.build(report)
        return report

    # ---- healthy: no retrieval at all -------------------------------------
    if is_healthy:
        # The mask reads a few percent of non-green on almost any real photograph -
        # leaf edge, vein shadow, a speck of soil. On a leaf the classifier called
        # healthy that is noise, not disease, so the band must not drive an action
        # urgency: a healthy grape leaf was reporting 8.1% "Mild" and telling the
        # farmer to act within the week.
        report["urgency"] = config.SEVERITY_URGENCY["Healthy"]
        report["severity_advice"] = config.SEVERITY_ADVICE["Healthy"]
        report["sections"]["summary"] = {
            "heading": "Assessment",
            "tier": "H",
            "passages": [],
            "note": (f"No disease detected ({report['confidence_pct']} confidence). "
                     "The leaf appears healthy. Continue routine monitoring and "
                     "preventive practices."),
        }
        report["plan"] = action_plan.build(report)
        return report

    # ---- low confidence: identify only, no treatment ----------------------
    if low_conf:
        alts = ", ".join(f"{a['name']} ({a['pct']})" for a in report["alternatives"])
        report["caveats"].insert(0, (
            f"Confidence {report['confidence_pct']} is below the "
            f"{gate * 100:.0f}% reliability threshold, so no treatment guidance is given."))
        report["sections"]["summary"] = {
            "heading": "Uncertain identification",
            "tier": "C",
            "passages": [],
            "note": (f"The model could not identify this confidently. Other candidates: "
                     f"{alts}. Please show a fresh, well-lit photograph to your nearest "
                     "Krishi Vigyan Kendra (KVK) before applying any treatment."),
        }
        report["plan"] = action_plan.build(report)
        return report

    # ---- retrieval, one block per requested section ------------------------
    name = (disease_title or label).lower()
    tiers, seen = [], set()

    for key, (section_type, template) in kb.section_queries().items():
        query = template.format(name=name, crop=crop)
        hits, tier = kb.retrieve(label, crop, query, section=section_type)
        passages = kb.to_passages(hits, seen=seen)      # dedup across blocks

        if not passages:
            # Everything this block found was already printed above, or nothing matched.
            tier = "C" if not hits else tier
            note = ("The knowledge base holds no further text for this section beyond "
                    "what is already shown above.") if hits else \
                   "No verified guidance available for this section."
        elif tier == "A":
            note = ""
        elif tier == "B":
            note = (f"No {name}-specific text exists in the knowledge base for this "
                    f"section. The passages below are general {crop} IPM guidance and "
                    "must be confirmed with an extension officer before use.")
        else:
            note = "No verified guidance available for this section."

        if passages:
            tiers.append(tier)

        heading = config.SECTION_HEADINGS[key]
        if tier == "B" and passages:
            heading += f" — general {crop} guidance"

        report["sections"][key] = {
            "heading": heading,
            "tier": tier,
            "passages": passages,
            "note": note,
        }
        # A passage whose section came from the content-rescue pass rather than a
        # heading is weaker evidence. Say so once per block instead of silently mixing.
        if any(p.get("rescued") for p in passages):
            note = (note + " " if note else "") + (
                "One or more passages here were matched to this section by content "
                "because the source document has no heading for it.")
            report["sections"][key]["note"] = note

        report["citations"] += [p["citation"] for p in passages]

    # Prevention now has its own rendered block (config.SECTION_QUERIES), so the plan
    # reuses those passages instead of running a second retrieval that would return
    # the same text under a different heading.
    prevention_sec = report["sections"].get("prevention", {})
    report["plan_sources"] = list(prevention_sec.get("passages", []))

    report["tier"] = "A" if "A" in tiers else ("B" if "B" in tiers else "C")
    report["citations"] = sorted(set(report["citations"]))

    # ---- look-alike differential -------------------------------------------
    # Retrieved with its own `seen` set: these passages describe OTHER diseases and
    # must never be mistaken for evidence about this one, so they are kept out of the
    # main dedup pool and rendered in their own block.
    differential_blocks = {}
    for other, sim in differential.look_alikes(label):
        o_crop, _, o_title = classifier.pretty(other)
        hits, _t = kb.retrieve(
            other, o_crop.lower(),
            f"distinguishing symptoms of {o_title.lower()} on {o_crop.lower()} leaves",
            section="symptom", k=config.DIFFERENTIAL_PASSAGES)
        passages = kb.to_passages(hits)[:config.DIFFERENTIAL_PASSAGES]
        if passages:
            differential_blocks[o_title] = passages
            report["differential"].append({
                "label": other, "name": o_title,
                "similarity": round(sim, 3),
                "passages": passages,
            })

    report["plan"] = action_plan.build(report)

    # ---- LLM prose, on top of the evidence, never instead of it -------------
    # The extractive sections above are already complete and cited. The advisory is
    # additive: if the model is unavailable or the audit rejects the draft, the report
    # is exactly what it would have been without it.
    blocks_for_llm = {config.SECTION_HEADINGS.get(k, k): v["passages"]
                      for k, v in report["sections"].items() if v.get("passages")}
    if blocks_for_llm:
        case = {
            "crop": crop_title,
            "disease": full_title,
            "confidence": report["confidence_pct"],
            "severity": (report["severity_display"]
                         if report["severity_percent"] is not None else "not measured"),
            "band": report["severity_level"],
            "urgency": report["urgency"],
        }
        report["advisory"] = advisor.write_advisory(
            case, blocks_for_llm, differential_blocks, report["caveats"])

    if report["tier"] != "A" or label in kb.fallback_classes():
        report["caveats"].append(
            "Parts of this report fall back to general crop guidance because the "
            "knowledge base has no disease-specific text. Please confirm with your "
            "nearest KVK before treating.")

    return report


def to_json(report):
    """Strip the base64 images down for API responses that only want the text."""
    out = {k: v for k, v in report.items() if k != "images"}
    out["images"] = {k: bool(v) for k, v in report["images"].items()}
    return out


def to_text(report):
    """Plain-text rendering, used for the download button."""
    L = ["PLANTCARE ADVISORY REPORT",
         f"Generated: {report['generated_at']}",
         f"Image: {report['filename']}",
         "=" * 70,
         f"CROP      : {report['crop']}",
         f"DIAGNOSIS : {report['disease']}  ({report['confidence_pct']} confidence)",
         f"SEVERITY  : {report['severity_level']}  {report['severity_display']}",
         f"TIER      : {report['tier']}",
         "=" * 70, ""]
    if report["severity_advice"]:
        L += ["SEVERITY GUIDANCE", "-" * 17, report["severity_advice"], ""]
    adv = report.get("advisory")
    if adv and adv.get("ok"):
        L += ["FIELD ADVISORY", "-" * 14]
        for sec in adv["sections"]:
            L += [sec["heading"].upper()]
            if sec.get("body"):
                L += [sec["body"]]
            L += [f"  - {b}" for b in sec.get("bullets", [])]
            L += [""]
        L += ["-" * 70, ""]

    blocks = () if (adv and adv.get("ok")) else (
        "summary", "symptoms", "cause", "reason", "precaution", "prevention")
    for key in blocks:
        sec = report["sections"].get(key)
        if not sec:
            continue
        L += [sec["heading"].upper(), "-" * len(sec["heading"])]
        if sec.get("note"):
            L += [sec["note"], ""]
        for p in sec["passages"]:
            L += [f"- {p['text']}  [{p['citation']}]"]
        L += [""]
    if report["caveats"]:
        L += ["IMPORTANT", "-" * 9] + [f"! {c}" for c in report["caveats"]] + [""]
    if adv and adv.get("ok") and adv.get("sources"):
        L += ["SOURCES", "-" * 7]
        L += [f"[S{x['n']}] {x['citation']}" for x in adv["sources"]]
    elif report["citations"]:
        L += ["SOURCES", "-" * 7, "  ".join(report["citations"])]
    return "\n".join(L)
