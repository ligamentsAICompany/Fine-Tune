# ── nokia_rag/server.py ──────────────────────────────────
# FastAPI backend — serves the UI and chat API

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from FlagEmbedding import FlagModel

import config
from indexer import load_or_build_index
from retriever import retrieve, _get_reranker
from llm import generate_answer

# ── App state ────────────────────────────────────────────

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Nokia fine-tuned BGE-Large...")
    model = FlagModel(
        config.MODEL_PATH,
        query_instruction_for_retrieval=config.QUERY_INSTRUCTION,
        use_fp16=True,
    )
    chunks, embeddings = load_or_build_index(model)
    print("Loading BGE Reranker (downloading on first run, ~570MB)...")
    _get_reranker()   # pre-warm so the first query isn't slow
    print("Reranker ready.")
    state["model"] = model
    state["chunks"] = chunks
    state["embeddings"] = embeddings
    state["history"] = []
    print(f"Ready — {len(chunks)} chunks indexed.")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)

# ── Serve static files ───────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── API ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    top_chunks, scores = retrieve(
        query,
        state["model"],
        state["chunks"],
        state["embeddings"],
    )

    answer = generate_answer(query, top_chunks, state["history"])

    sources = [
        {"text": chunk[:400] + ("..." if len(chunk) > 400 else ""), "score": round(score, 3)}
        for chunk, score in zip(top_chunks, scores)
    ]

    return {"answer": answer, "sources": sources}


@app.post("/api/reset")
async def reset():
    state["history"].clear()
    return {"status": "ok"}
