import json
import os

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class Query(BaseModel):
    vector: list[float]
    k: int = 5


app = FastAPI()
index = None
ids = []
passages = {}  # dict: id → text


@app.on_event("startup")
def load_index_and_passages():
    global index, ids, passages

    embeddings = np.load("/data/embeddings/embeddings.npy")
    with open("/data/embeddings/ids.json", "r") as f:
        ids = json.load(f)

    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    cpu_index.add(embeddings)

    index = cpu_index

    print(f"Loaded FAISS index with {index.ntotal} vectors.")

    raw_dir = "/data/raw"
    for fname in os.listdir(raw_dir):
        if not (fname.startswith("text-corpus_") and fname.endswith(".jsonl")):
            continue
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                passages[obj["id"]] = obj["passage"]

    print(f"Loaded {len(passages)} passages from raw JSONL.")


@app.post("/search")
def search(q: Query):
    if index is None:
        raise RuntimeError("FAISS index not loaded")
    vec = np.array([q.vector], dtype="float32")
    faiss.normalize_L2(vec)
    similarity_scores, neighbor_ids = index.search(vec, q.k)

    results = []
    for rank, idx in enumerate(neighbor_ids[0]):
        doc_id = ids[idx]
        score = float(similarity_scores[0][rank])
        text = passages.get(doc_id, "")
        results.append({"id": doc_id, "score": score, "text": text})

    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok", "vectors": index.ntotal}
