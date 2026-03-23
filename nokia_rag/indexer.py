# ── nokia_rag/indexer.py ────────────────────────────────
# Handles PDF extraction, chunking, embedding, and caching

import hashlib
import re
import pickle
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import FlagModel
from rich.console import Console

import config

console = Console()


def extract_text(pdf_path: str) -> str:
    """Extract and clean text from Nokia PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # Fix common PDF artifacts
    text = re.sub(r'-\n', '', text)           # Hyphenated line breaks
    text = re.sub(r'\f', ' ', text)            # Form feeds
    text = re.sub(r'\n{3,}', '\n\n', text)    # Excess blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)    # Excess whitespace
    return text


def build_chunks(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    raw = splitter.split_text(text)
    # Deduplicate and filter short chunks
    chunks = list(dict.fromkeys(c.strip() for c in raw if len(c.strip()) >= config.MIN_CHUNK_LEN))
    return chunks


def embed_chunks(chunks: list[str], model: FlagModel) -> np.ndarray:
    """Encode chunks and normalize embeddings."""
    console.print(f"[yellow]Encoding {len(chunks)} chunks...[/yellow]")
    embeddings = model.encode(chunks, batch_size=32)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def get_pdf_hash(pdf_path: str) -> str:
    """Compute and return the MD5 hex digest of a PDF file."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_build_index(model: FlagModel) -> tuple[list[str], np.ndarray]:
    """
    Load cached index if it exists AND the source PDF is unchanged,
    otherwise build from scratch.

    Cache format (NEW — dict):
        {
            "hash": "<md5-hex>",   # MD5 of the PDF at index-build time
            "chunks": [...],        # list of text chunks
            "embeddings": ndarray   # normalised embeddings
        }

    Old format (tuple) is detected and treated as stale → auto-rebuild.
    NOTE: After first run with this version, the old nokia_index.pkl is
    replaced automatically. You may also delete it manually before starting.
    Returns (chunks, embeddings).
    """
    if not Path(config.PDF_PATH).exists():
        raise FileNotFoundError(
            f"PDF not found at '{config.PDF_PATH}'. "
            f"Update PDF_PATH in config.py or set NOKIA_PDF_PATH in .env"
        )

    current_hash = get_pdf_hash(config.PDF_PATH)
    cache_path   = Path(config.INDEX_CACHE)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        # Validate: must be new dict format with matching hash
        if (
            isinstance(cached, dict)
            and "hash" in cached
            and cached["hash"] == current_hash
        ):
            chunks     = cached["chunks"]
            embeddings = cached["embeddings"]
            console.print("[green]✅ Loading cached index...[/green]")
            console.print(f"[green]   {len(chunks)} chunks loaded (PDF unchanged)[/green]")
            return chunks, embeddings

        # Cache is stale — explain why before rebuilding
        if not isinstance(cached, dict) or "hash" not in cached:
            console.print("[yellow]⚠️  Old cache format detected — rebuilding index...[/yellow]")
        else:
            console.print("[yellow]⚠️  PDF has changed since last index build — rebuilding...[/yellow]")

    else:
        console.print("[yellow]Building index from PDF (first-time only)...[/yellow]")

    # ── Build from scratch ────────────────────────────────────────────────────
    text       = extract_text(config.PDF_PATH)
    chunks     = build_chunks(text)
    embeddings = embed_chunks(chunks, model)

    # Cache to disk in NEW dict format (includes PDF hash for invalidation)
    with open(cache_path, "wb") as f:
        pickle.dump({"hash": current_hash, "chunks": chunks, "embeddings": embeddings}, f)

    console.print(f"[green]✅ Index built and cached: {len(chunks)} chunks[/green]")
    return chunks, embeddings
