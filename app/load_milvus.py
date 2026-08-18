"""Load the knowledge base into a running Milvus Standalone server.

Only needed for the Docker path. The embedded and numpy backends read the
artifacts directly and need no loading step.

    docker compose up -d
    python load_milvus.py                      # defaults to localhost:19530
    set PLANTCARE_MILVUS_URI=http://localhost:19530   &&  python app.py

Reads artifacts/chunks.csv + artifacts/embeddings.npy - the same files the numpy
backend uses - so there is no separate export step.
"""

import os
import sys

import numpy as np
import pandas as pd
from pymilvus import DataType, MilvusClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import knowledge_base  # noqa: E402

URI = os.getenv("PLANTCARE_MILVUS_URI", "http://localhost:19530")
DIM = 384
BATCH = 256


def main():
    chunks_csv = config.ARTIFACT_DIR / "chunks.csv"
    embed_npy = config.ARTIFACT_DIR / "embeddings.npy"
    for p in (chunks_csv, embed_npy):
        if not p.exists():
            sys.exit(f"missing {p} - copy it out of plantcare_kb.zip first")

    frame = pd.read_csv(chunks_csv).fillna("")
    vecs = np.load(embed_npy).astype(np.float32)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    if len(frame) != len(vecs):
        sys.exit(f"{len(frame)} chunks but {len(vecs)} vectors - mismatched files")
    print(f"{len(frame)} chunks, {vecs.shape[1]}d")

    client = MilvusClient(uri=URI, token=os.getenv("PLANTCARE_MILVUS_TOKEN") or None)
    print("connected to", URI)

    if client.has_collection(knowledge_base.COLLECTION):
        print("dropping existing collection")
        client.drop_collection(knowledge_base.COLLECTION)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=4000, enable_analyzer=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("crop", DataType.VARCHAR, max_length=16)
    schema.add_field("disease", DataType.VARCHAR, max_length=64)
    schema.add_field("all_diseases", DataType.VARCHAR, max_length=512)
    schema.add_field("section_type", DataType.VARCHAR, max_length=16)
    schema.add_field("source_pdf", DataType.VARCHAR, max_length=128)
    schema.add_field("page_no", DataType.INT32)

    use_bm25 = True
    try:
        from pymilvus import Function, FunctionType
        schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                     input_field_names=["text"],
                                     output_field_names=["sparse"]))
    except Exception as exc:                        # noqa: BLE001
        print("BM25 function unavailable, dense only:", str(exc)[:120])
        use_bm25 = False

    idx = client.prepare_index_params()
    idx.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
    if use_bm25:
        idx.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                      metric_type="BM25")

    client.create_collection(knowledge_base.COLLECTION, schema=schema, index_params=idx)
    print("collection created | bm25:", use_bm25)

    rows = []
    for i, r in frame.reset_index(drop=True).iterrows():
        rows.append({"text": str(r["text"])[:3990], "dense": vecs[i].tolist(),
                     "crop": str(r["crop"]), "disease": str(r["disease"]),
                     "all_diseases": str(r["all_diseases"])[:500],
                     "section_type": str(r["section_type"]),
                     "source_pdf": str(r["source_pdf"]), "page_no": int(r["page_no"])})

    for i in range(0, len(rows), BATCH):
        client.insert(knowledge_base.COLLECTION, rows[i:i + BATCH])
    client.flush(knowledge_base.COLLECTION)
    client.load_collection(knowledge_base.COLLECTION)

    print("loaded:", client.get_collection_stats(knowledge_base.COLLECTION))
    print(f"\nNow start the app with:\n"
          f"  PLANTCARE_MILVUS_URI={URI} python app.py        (WSL / macOS)\n"
          f"  set PLANTCARE_MILVUS_URI={URI} && python app.py  (Windows cmd)")


if __name__ == "__main__":
    main()
