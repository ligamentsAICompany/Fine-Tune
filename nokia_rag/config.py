# ── nokia_rag/config.py ─────────────────────────────────
# Central config — change these to match your setup

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # picks up root .env (OLLAMA_HOST, NOKIA_PDF_PATH, etc.)

# ── Paths (absolute — works from any working directory) ──────────────────────
_HERE = Path(__file__).resolve().parent          # .../nokia_rag/
_ROOT = _HERE.parent                             # repo root

# Override PDF path via env var NOKIA_PDF_PATH; default to source_files/
_pdf_env = os.getenv("NOKIA_PDF_PATH")
PDF_PATH = str(Path(_pdf_env).resolve()) if _pdf_env else str(
    _ROOT / "source_files" / "Alarms_Guide_3HH13538AAAATCZZA19.pdf"
)

MODEL_PATH  = str(_ROOT / "nokia-isam-bge-large")   # Fine-tuned BGE-Large
INDEX_CACHE = str(_HERE / "nokia_index.pkl")          # Cached chunk embeddings

# ── LLM ──────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 150
MIN_CHUNK_LEN = 150

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20   # candidates fetched before reranking
TOP_K = 5              # final chunks passed to LLM after reranking

# ── BGE query instruction (required for BGE models) ──────────────────────────
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
