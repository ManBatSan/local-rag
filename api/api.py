import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ─── Config ────────────────────────────────────────────────────────────────────
VECTORSTORE_URL = os.getenv("VECTORSTORE_URL", "http://vectorstore:8001")
MODELSERVER_URL = os.getenv("MODELSERVER_URL", "http://model-server:8002")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda")
TOP_K = int(os.getenv("TOP_K", "5"))
# ────────────────────────────────────────────────────────────────────────────────

app = FastAPI()
embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
client = httpx.AsyncClient(timeout=60.0)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    q_vec = embedder.encode(req.question, normalize_embeddings=True).tolist()

    try:
        vs_resp = await client.post(
            f"{VECTORSTORE_URL}/search",
            json={"vector": q_vec, "k": TOP_K},
        )
        vs_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Vectorstore error: {e}")

    results = vs_resp.json()["results"]
    passages = []
    ids = []
    for r in results:
        ids.append(r["id"])
        passages.append(r.get("text", ""))

    prompt = "You are a helpful assistant. Use the following excerpts to answer the question. "
    "When answering quote the parts of the excerpts that you used to base your answer. "
    "If you have prior knowledge contradicting the excerpts please just use the excerpts. "
    "First answer the question shortly, "
    "then highlight only the excerpts that you used to base your answer \n\n"
    for i, p in enumerate(passages, 1):
        prompt += f"[{i}] {p}\n\n"
    prompt += f"Question: {req.question}\nAnswer:"

    try:
        ms_resp = await client.post(
            f"{MODELSERVER_URL}/generate",
            json={"prompt": prompt, "max_length": 256, "temperature": 0.7},
        )
        ms_resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Model-server error: {e}")

    answer = ms_resp.json().get("generated_text", "").strip()

    return ChatResponse(answer=answer, sources=[str(i) for i in ids])
