"""Knowledge base facade.

Picks a backend, runs tiered retrieval, and converts hits into citable passages.
This is the only module report.py talks to.

Backend selection (PLANTCARE_KB_BACKEND=numpy|milvus forces one):
    1. PLANTCARE_MILVUS_URI set          -> Milvus server
    2. plantcare.db present AND pymilvus importable -> Milvus Lite
    3. chunks.csv + embeddings.npy       -> numpy  (the Windows path)

Tiers
    A  chunks tagged with the predicted disease      -> full advisory
    B  no disease chunks, but crop text exists       -> labelled "general"
    C  nothing                                       -> report says so
"""

import json
import re
from pathlib import Path

import numpy as np

import config
import kb_backends

# ----------------------------------------------------------------- manifest
MANIFEST = (json.loads(Path(config.KB_MANIFEST).read_text(encoding="utf-8"))
            if Path(config.KB_MANIFEST).exists() else {})

COLLECTION = MANIFEST.get("collection", "plantcare_kb")

# Notebook 02 writes this key as "embed_model". Reading only "embedding_model" made the
# app silently fall back to bge-SMALL (384-d) while embeddings.npy holds bge-BASE
# vectors (768-d) - a dimension mismatch that surfaces as a crash or, worse, as
# meaningless similarities. Both spellings are accepted and the dimension is asserted
# against the stored vectors at startup.
EMB_MODEL = (MANIFEST.get("embed_model")
             or MANIFEST.get("embedding_model")
             or "BAAI/bge-base-en-v1.5")
EMB_DIM_EXPECTED = {"BAAI/bge-base-en-v1.5": 768, "BAAI/bge-small-en-v1.5": 384,
                    "BAAI/bge-large-en-v1.5": 1024}.get(EMB_MODEL)
WANT_BM25 = bool(MANIFEST.get("hybrid_bm25", MANIFEST.get("sparse_in_milvus", False)))
_FALLBACK = set(MANIFEST.get("fallback_classes", []))


def fallback_classes():
    """Classes the KB build flagged as THIN or MISSING."""
    return set(_FALLBACK)


def manifest():
    return dict(MANIFEST)


# ----------------------------------------------------------------- backend
def _pick_backend():
    forced = config.KB_BACKEND

    if forced != "numpy":
        if config.MILVUS_URI:
            return kb_backends.MilvusBackend(config.MILVUS_URI, COLLECTION, WANT_BM25)
        if Path(config.KB_DB).exists():
            try:
                return kb_backends.MilvusBackend(str(config.KB_DB), COLLECTION, WANT_BM25)
            except Exception as exc:              # noqa: BLE001
                if forced == "milvus":
                    raise
                # Expected on Windows: milvus-lite ships no wheel there.
                print(f"[kb] Milvus unavailable ({type(exc).__name__}: "
                      f"{str(exc)[:100]}) - using numpy backend")
        elif forced == "milvus":
            raise FileNotFoundError(f"{config.KB_DB} not found but backend forced to milvus")

    if not (Path(config.KB_CHUNKS).exists() and Path(config.KB_VECTORS).exists()):
        raise FileNotFoundError(
            f"No usable knowledge base. Need either {config.KB_DB} (Milvus Lite) or "
            f"{config.KB_CHUNKS} + {config.KB_VECTORS} (numpy). Export them from "
            "notebook 02 into artifacts/.")
    return kb_backends.NumpyBackend(config.KB_CHUNKS, config.KB_VECTORS)


backend = _pick_backend()


def _fingerprint():
    """Print what was actually loaded.

    The commonest failure after rebuilding the KB is forgetting to copy the new
    artifacts, then concluding the rebuild changed nothing. These four lines make a
    stale knowledge base impossible to miss at startup.
    """
    st = backend.stats()
    types = set(MANIFEST.get("section_types", []) or st.get("section_types", []))
    vocab = ("v2 (seven sections)"
             if {"pathogen", "transmission", "treatment"} <= types else
             "LEGACY (four sections)")
    print(f"[kb] backend    : {st.get('backend')} | {st.get('hybrid')}")
    print(f"[kb] chunks     : {st.get('chunks')} from "
          f"{MANIFEST.get('n_documents', '?')} documents")
    print(f"[kb] vocabulary : {vocab}")
    print(f"[kb] embedder   : {EMB_MODEL}"
          + (f" ({EMB_DIM_EXPECTED}-d)" if EMB_DIM_EXPECTED else ""))

    # Catch a mismatched vector file before the first query rather than after.
    vec = Path(config.KB_VECTORS)
    if vec.exists() and EMB_DIM_EXPECTED:
        got = int(np.load(vec, mmap_mode="r").shape[1])
        if got != EMB_DIM_EXPECTED:
            raise RuntimeError(
                f"embeddings.npy is {got}-d but kb_manifest.json says the KB was built "
                f"with {EMB_MODEL} ({EMB_DIM_EXPECTED}-d). One of the two artifacts is "
                f"stale - re-copy both from the same notebook 02 run.")
    if types:
        print(f"[kb] sections   : {', '.join(sorted(types))}")
    if vocab.startswith("LEGACY"):
        print("[kb] WARNING this is the old knowledge base. Copy chunks.csv, "
              "embeddings.npy,\n"
              "     plantcare.db/, kb_manifest.json and knowledge_graph.json from "
              "plantcare_kb.zip\n"
              "     into artifacts/ - 'cause' and 'reason' share one pool until you do.")
    return vocab


KB_VOCAB = _fingerprint()


# ----------------------------------------------------------------- embedder
_embedder = None


def _encode(text):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer   # noqa: PLC0415
        _embedder = SentenceTransformer(EMB_MODEL)
        print(f"[kb] embedder: {EMB_MODEL}")
    return _embedder.encode([text], normalize_embeddings=True)[0]


# ----------------------------------------------------------------- graph
class KnowledgeGraph:
    """Gazetteer graph, used ONLY to expand the lexical (BM25) query.

    The previous version fed the expansion to the dense encoder too. That is
    counter-productive: expansion produces an unordered bag of words, and BGE is a
    sentence encoder whose quality depends on the sentence still being a sentence.
    Dense now gets the original query; BM25 gets the expanded one, where extra
    keywords genuinely help.
    """

    def __init__(self, path):
        self.graph = None
        p = Path(path) if path else None
        if not (p and p.exists()):
            return
        try:
            import networkx as nx                    # noqa: PLC0415
            data = json.loads(p.read_text(encoding="utf-8"))
            try:
                self.graph = nx.node_link_graph(data, edges="edges")
            except TypeError:                        # networkx < 3.4
                self.graph = nx.node_link_graph(data)
            print(f"[kb] graph: {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")
        except Exception as exc:                     # noqa: BLE001
            print("[kb] graph unavailable:", exc)

    def expand(self, query, disease=None):
        if self.graph is None or disease not in (self.graph or {}):
            return query
        extra = []
        for nbr in self.graph.neighbors(disease):
            extra.append(str(nbr))
            for nbr2 in self.graph.neighbors(nbr):
                extra.append(str(nbr2))
        if not extra:
            return query
        # Query first, keywords appended - order is stable and the query stays intact.
        seen, tail = set(), []
        for t in extra:
            t = t.replace("_", " ")
            if t.lower() not in seen:
                seen.add(t.lower())
                tail.append(t)
        return query + " " + " ".join(tail[:12])


kg = KnowledgeGraph(config.KNOWLEDGE_GRAPH)


# ----------------------------------------------------------------- retrieval
def _search(query, k, crop=None, section=None, disease=None):
    qv = _encode(query)
    bm25_text = kg.expand(query, disease)
    return backend.search(qv, query, k, crop=crop, section=section,
                          disease=disease, bm25_text=bm25_text)


def retrieve(disease, crop, query, section=None, k=4):
    """Returns (hits, tier). Tier A disease-specific, B crop-general, C nothing."""
    if section:
        hits = _search(query, k, crop=crop, section=section, disease=disease)
        if hits:
            return hits, "A"

    hits = _search(query, k, crop=crop, disease=disease)
    if hits:
        return hits, "A"

    if section:
        hits = _search(query, k, crop=crop, section=section)
        if hits:
            return hits, "B"

    hits = _search(query, k, crop=crop)          # crop-only fallback
    if hits:
        return hits, "B"

    return [], "C"


retrieve_tiered = retrieve      # alias kept for callers using the old name


# ----------------------------------------------------------------- passages
def _tidy(text, limit=420):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def to_passages(hits, limit=420, seen=None):
    """Hits -> JSON-safe passage dicts with page citations.

    Every value is a native Python type. numpy scalars (page_no arrives as
    numpy.int64 from pandas) are not JSON-serialisable and used to break
    /api/analyse at the very last step.

    Pass a `seen` set to de-duplicate across report blocks - the "cause" and
    "reason" blocks draw on the same small pool and would otherwise repeat.
    """
    out = []
    for h in hits:
        text = _tidy(h.get("text", ""), limit)
        if not text:
            continue
        key = text[:120].lower()
        if seen is not None:
            if key in seen:
                continue
            seen.add(key)
        page = h.get("page_no", 0)
        page = int(page.item()) if isinstance(page, np.generic) else int(page)
        out.append({
            "text": text,
            "citation": f"{h.get('source_pdf', 'source')} p.{page}",
            "source_pdf": str(h.get("source_pdf", "")),
            "page_no": page,
            "section": str(h.get("section_type", "")),
            "section_title": str(h.get("section_title", "")),
            # True when notebook 02's content-rescue pass assigned this chunk's section
            # rather than a heading. Surfaced so a reader can weigh it accordingly.
            "rescued": str(h.get("section_source", "heading")) == "content",
            "score": round(float(h.get("score", 0.0)), 5),
        })
    return out


def section_queries():
    """Pick the mapping that matches the KB actually loaded.

    A new-vocabulary KB (seven section types) and the legacy four-type one need
    different mappings. Guessing wrong silently empties the cause/reason blocks, so
    it is read from the manifest rather than assumed.
    """
    types = set(MANIFEST.get("section_types", []))
    if {"pathogen", "transmission", "treatment"} <= types:
        return config.SECTION_QUERIES
    if types:
        print("[kb] legacy four-section KB detected - using SECTION_QUERIES_LEGACY. "
              "Rebuild with the rewritten notebook 02 for distinct cause/reason blocks.")
    return config.SECTION_QUERIES_LEGACY


def health():
    """What /health reports about the KB."""
    info = backend.stats()
    info["embedding_model"] = EMB_MODEL
    info["flagged_classes"] = sorted(_FALLBACK)
    info["vocabulary"] = KB_VOCAB
    info["section_types"] = MANIFEST.get("section_types", [])
    info["n_documents"] = MANIFEST.get("n_documents")
    info["chunks_by_section_source"] = MANIFEST.get("chunks_by_section_source", {})
    info["section_mapping"] = {k: v[0] for k, v in section_queries().items()}
    return info
