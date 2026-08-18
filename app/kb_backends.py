"""Retrieval backends.

Two implementations, one interface. Both return a plain list of dicts with the
same keys, so nothing above this file knows or cares which one is running:

    {text, crop, disease, all_diseases, section_type, source_pdf, page_no, score}

`NumpyBackend` is the portable one - it needs only chunks.csv and embeddings.npy
and runs anywhere, including Windows, where milvus-lite has no wheel. It carries
its own BM25 so hybrid search works there too.

`MilvusBackend` talks to Milvus Lite (a local .db directory) or a Milvus server
(set PLANTCARE_MILVUS_URI). Milvus Lite's sparse-vector support is limited, so
hybrid is probed once at startup rather than assumed - see `hybrid_ready`.

Both apply the disease filter as an EXACT match on the comma-separated
`all_diseases` field. Milvus can only express `like "%x%"`, so its results are
re-filtered in Python; a class name that happens to be a prefix of another can
never leak through.
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

import config

CORE_FIELDS = ["text", "crop", "disease", "all_diseases", "section_type",
               "source_pdf", "page_no"]
# Written by the rewritten notebook 02. Absent from older builds, so they are
# requested only when the collection actually declares them.
EXTRA_FIELDS = ["section_title", "keywords", "section_source"]
FIELDS = CORE_FIELDS + EXTRA_FIELDS

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return _TOKEN.findall(str(text).lower())


def has_disease(all_diseases, disease):
    """Exact membership in the comma-separated tag list - not a substring test."""
    if not disease:
        return True
    return disease in {d.strip() for d in str(all_diseases).split(",") if d.strip()}


def rrf(*rankings, k=None):
    """Reciprocal rank fusion. Each ranking is an ordered list of row indices."""
    k = config.RRF_K if k is None else k
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True), fused



# The two notebook-02 generations name the same columns differently. Normalising once
# here beats sprinkling `if "page_no" in df` through every reader - a missing alias used
# to surface as KeyError deep inside search().
COLUMN_ALIASES = {
    "page":        "page_no",     # rewritten NB02 -> original app vocabulary
    "doc":         "source_pdf",
    "document":    "source_pdf",
    "source":      "source_pdf",
    "section":     "section_type",
}


def normalise_columns(df):
    for new, old in COLUMN_ALIASES.items():
        if new in df.columns and old not in df.columns:
            df[old] = df[new]
    if "all_diseases" not in df.columns and "disease" in df.columns:
        df["all_diseases"] = df["disease"].astype(str)
    if "page_no" not in df.columns:
        df["page_no"] = 0
    if "source_pdf" not in df.columns:
        df["source_pdf"] = ""
    return df


# ---------------------------------------------------------------- numpy
class NumpyBackend:
    """Portable dense + BM25 retrieval straight out of chunks.csv."""

    name = "numpy"
    hybrid_ready = True

    def __init__(self, chunks_path, vectors_path):
        self.df = normalise_columns(pd.read_csv(chunks_path).fillna(""))
        vecs = np.load(vectors_path).astype(np.float32)
        if len(vecs) != len(self.df):
            raise ValueError(
                f"embeddings.npy has {len(vecs)} rows but chunks.csv has "
                f"{len(self.df)} - they came from different KB builds")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vecs = vecs / norms
        self._build_bm25()

    def _build_bm25(self):
        # The rewritten notebook 02 exports `search_text` - the chunk plus its pathogen
        # binomials, aliases and chemical names. That is the field BM25 is supposed to
        # index; matching on `text` alone throws the keyword enrichment away, because a
        # query saying "black measles" never meets "Phaeomoniella chlamydospora".
        self.bm25_col = "search_text" if "search_text" in self.df.columns else "text"
        self.docs = [tokenize(t) for t in self.df[self.bm25_col]]
        self.tf = [Counter(d) for d in self.docs]
        self.dl = np.array([max(1, len(d)) for d in self.docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean())
        n_docs = len(self.docs)
        doc_freq = Counter()
        for d in self.docs:
            doc_freq.update(set(d))
        self.idf = {
            term: math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))
            for term, n in doc_freq.items()
        }

    def _candidates(self, crop=None, section=None, disease=None):
        mask = np.ones(len(self.df), dtype=bool)
        if crop:
            mask &= (self.df["crop"].astype(str) == crop).to_numpy()
        if section:
            mask &= (self.df["section_type"].astype(str) == section).to_numpy()
        if disease:
            # The rewritten notebook 02 tags one disease per chunk (`disease`); the
            # older multi-label build wrote a semicolon list (`all_diseases`). Accept
            # whichever the loaded chunks.csv actually has - keying on the missing one
            # raised KeyError and took every query down.
            if "all_diseases" in self.df.columns:
                mask &= self.df["all_diseases"].map(
                    lambda v: has_disease(v, disease)).to_numpy()
            elif "disease" in self.df.columns:
                mask &= (self.df["disease"].astype(str) == disease).to_numpy()
        return np.flatnonzero(mask)

    def _bm25(self, q_tokens, idx):
        k1, b = config.BM25_K1, config.BM25_B
        scores = np.zeros(len(idx), dtype=np.float32)
        for term in set(q_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for j, i in enumerate(idx):
                f = self.tf[i].get(term, 0)
                if f:
                    denom = f + k1 * (1 - b + b * self.dl[i] / self.avgdl)
                    scores[j] += idf * f * (k1 + 1) / denom
        return scores

    def search(self, query_vec, query_text, k, crop=None, section=None,
               disease=None, bm25_text=None):
        idx = self._candidates(crop=crop, section=section, disease=disease)
        if idx.size == 0:
            return []

        dense = self.vecs[idx] @ np.asarray(query_vec, dtype=np.float32)
        dense_rank = [idx[i] for i in np.argsort(-dense)]

        sparse = self._bm25(tokenize(bm25_text or query_text), idx)
        if sparse.any():
            sparse_rank = [idx[i] for i in np.argsort(-sparse)]
            order, fused = rrf(dense_rank, sparse_rank)
        else:                                   # no lexical overlap at all
            order, fused = dense_rank, {idx[i]: float(dense[i]) for i in range(len(idx))}

        out = []
        for row in order[:k]:
            r = self.df.iloc[int(row)]
            hit = {f: str(r[f]) for f in CORE_FIELDS if f in self.df.columns}
            try:
                hit["page_no"] = int(float(r["page_no"]))
            except (TypeError, ValueError):
                hit["page_no"] = 0
            for f in EXTRA_FIELDS:
                if f in self.df.columns:
                    hit[f] = str(r[f])
            hit["score"] = float(fused.get(row, 0.0))
            out.append(hit)
        return out

    def stats(self):
        return {"backend": "numpy", "chunks": int(len(self.df)),
                "hybrid": f"dense + own BM25 over `{self.bm25_col}` (RRF)",
                "section_types": sorted(self.df.section_type.unique().tolist())
                if "section_type" in self.df.columns else []}


# ---------------------------------------------------------------- milvus
class MilvusBackend:
    """Milvus Lite (local .db) or a Milvus server."""

    name = "milvus"

    def __init__(self, uri, collection, want_bm25=True):
        from pymilvus import MilvusClient           # noqa: PLC0415
        self.client = MilvusClient(uri)
        self.collection = collection
        if self.client.has_collection(collection):
            self.client.load_collection(collection)
        self.fields = self._available_fields()
        self.hybrid_ready = self._probe(want_bm25)

    def _available_fields(self):
        """Older collections lack section_title/keywords/section_source. Requesting a
        field the schema does not declare fails the whole search, so intersect first."""
        try:
            declared = {f["name"] for f in
                        self.client.describe_collection(self.collection)["fields"]}
        except Exception:                           # noqa: BLE001
            return list(CORE_FIELDS)
        return [f for f in FIELDS if f in declared] or list(CORE_FIELDS)

    def _probe(self, want_bm25):
        """Actually attempt one hybrid search instead of assuming it works."""
        if not want_bm25:
            self.hybrid_note = "manifest says hybrid_bm25=false"
            return False
        try:
            from pymilvus import AnnSearchRequest, RRFRanker   # noqa: PLC0415
            # Read the dimension from the schema. It was hardcoded to 384, so a
            # collection built with bge-BASE (768-d) failed this probe every time and
            # silently dropped to dense-only retrieval - a real loss of recall that
            # looked like a working system.
            dim = 768
            try:
                for f in self.client.describe_collection(self.collection)["fields"]:
                    if f["name"] == "dense":
                        dim = int(f.get("params", {}).get("dim", dim))
                        break
            except Exception:                        # noqa: BLE001
                pass
            probe = [0.0] * dim
            probe[0] = 1.0
            reqs = [AnnSearchRequest([probe], "dense", {"metric_type": "COSINE"}, limit=1),
                    AnnSearchRequest(["leaf spot"], "sparse", {"metric_type": "BM25"}, limit=1)]
            self.client.hybrid_search(self.collection, reqs, ranker=RRFRanker(),
                                      limit=1, output_fields=["text"])
            self.hybrid_note = "dense + BM25 with RRF"
            return True
        except Exception as exc:                    # noqa: BLE001
            self.hybrid_note = (f"hybrid unavailable, dense-only "
                                f"({type(exc).__name__}: {str(exc)[:120]})")
            return False

    # Not a @staticmethod any more: which disease field to filter on depends on the
    # collection's own schema, so this needs self.fields.
    def _expr(self, crop=None, section=None, disease=None):
        clauses = []
        if disease:
            # Match the field the collection actually has (see _candidates).
            if "all_diseases" in set(self.fields or []):
                clauses.append(f'all_diseases like "%{disease}%"')
            else:
                clauses.append(f'disease == "{disease}"')
        if crop:
            clauses.append(f'crop == "{crop}"')
        if section:
            clauses.append(f'section_type == "{section}"')
        return " and ".join(clauses)

    def search(self, query_vec, query_text, k, crop=None, section=None,
               disease=None, bm25_text=None):
        expr = self._expr(crop, section, disease)
        over = k * 3                       # over-fetch, exact filter drops some
        raw = None

        if self.hybrid_ready:
            try:
                from pymilvus import AnnSearchRequest, RRFRanker   # noqa: PLC0415
                reqs = [
                    AnnSearchRequest([list(query_vec)], "dense",
                                     {"metric_type": "COSINE"}, limit=over, expr=expr),
                    AnnSearchRequest([bm25_text or query_text], "sparse",
                                     {"metric_type": "BM25"}, limit=over, expr=expr),
                ]
                raw = self.client.hybrid_search(self.collection, reqs,
                                                ranker=RRFRanker(k=config.RRF_K),
                                                limit=over, output_fields=self.fields)[0]
            except Exception:                       # noqa: BLE001
                self.hybrid_ready = False
                self.hybrid_note += " (failed at query time, fell back to dense)"

        if raw is None:
            raw = self.client.search(self.collection, data=[list(query_vec)],
                                     limit=over, filter=expr, output_fields=self.fields,
                                     search_params={"metric_type": "COSINE"})[0]

        out = []
        for h in raw:
            e = h.get("entity", h)
            if disease and "all_diseases" in e and not has_disease(
                    e.get("all_diseases", ""), disease):
                continue                            # exact match, not `like`
            hit = {f: str(e.get(f, "")) for f in self.fields if f != "page_no"}
            hit["page_no"] = int(e.get("page_no", 0))
            hit["score"] = float(h.get("distance", h.get("score", 0.0)) or 0.0)
            out.append(hit)
            if len(out) >= k:
                break
        return out

    def stats(self):
        try:
            n = self.client.get_collection_stats(self.collection).get("row_count")
        except Exception:                           # noqa: BLE001
            n = None
        return {"backend": "milvus", "chunks": n, "hybrid": self.hybrid_note,
                "output_fields": self.fields}

    def close(self):
        try:
            self.client.close()
        except Exception:                           # noqa: BLE001
            pass
