"""Rebuild artifacts/plantcare.db from kb_chunks.json + embeddings.npy."""
import json, shutil, os, sys
from pathlib import Path
import numpy as np
from pymilvus import MilvusClient, DataType

ART = Path("artifacts"); DB = ART / "plantcare.db"
payload = json.loads((ART / "kb_chunks.json").read_text())
chunks = payload["chunks"]; dense = np.load(ART / "embeddings.npy").astype(np.float32)
assert len(chunks) == len(dense), (len(chunks), len(dense))

if DB.is_dir(): shutil.rmtree(DB)
elif DB.exists(): DB.unlink()

c = MilvusClient(uri=str(DB))
s = c.create_schema(auto_id=False, enable_dynamic_field=False)
s.add_field("id", DataType.INT64, is_primary=True)
s.add_field("text", DataType.VARCHAR, max_length=4000)
s.add_field("search_text", DataType.VARCHAR, max_length=6000)
s.add_field("disease", DataType.VARCHAR, max_length=64)
s.add_field("all_diseases", DataType.VARCHAR, max_length=512)
s.add_field("crop", DataType.VARCHAR, max_length=32)
s.add_field("section_type", DataType.VARCHAR, max_length=32)
s.add_field("source_pdf", DataType.VARCHAR, max_length=256)
s.add_field("page_no", DataType.INT64)
s.add_field("dense", DataType.FLOAT_VECTOR, dim=int(dense.shape[1]))
ix = c.prepare_index_params()
ix.add_index(field_name="dense", index_type="FLAT", metric_type="IP")
c.create_collection("plantcare_kb", schema=s, index_params=ix)
c.insert("plantcare_kb", [{
    "id": int(k["id"]), "text": k["text"][:4000], "search_text": k["search_text"][:6000],
    "disease": k["disease"] or "", "all_diseases": k["disease"] or "",
    "crop": k["crop"] or "", "section_type": k["section_type"],
    "source_pdf": str(k["doc"])[:256], "page_no": int(k["page"]),
    "dense": dense[i].tolist()} for i, k in enumerate(chunks)])
c.flush("plantcare_kb"); c.load_collection("plantcare_kb")
print("rebuilt:", c.get_collection_stats("plantcare_kb"))
print("size:", sum(f.stat().st_size for f in DB.rglob("*") if f.is_file())/1024, "KB")
