# ── nokia_rag/retriever.py ──────────────────────────────
# Handles query embedding, candidate retrieval, and cross-encoder reranking

import numpy as np
from FlagEmbedding import FlagModel, FlagReranker
import config

# Lazy-loaded reranker — initialized once on first retrieve() call
_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


def retrieve(
    query: str,
    model: FlagModel,
    chunks: list[str],
    embeddings: np.ndarray,
) -> tuple[list[str], list[float]]:
    """
    Two-stage retrieval:
      1. Bi-encoder: cosine similarity to fetch TOP_K_RETRIEVAL candidates
      2. Cross-encoder (BGE reranker): rerank candidates, return top TOP_K
    """
    # ── Stage 1: bi-encoder candidate fetch ──
    q_emb = model.encode_queries([query])
    q_emb = q_emb / np.linalg.norm(q_emb)

    sims = embeddings @ q_emb[0]
    candidate_idx = np.argsort(sims)[::-1][: config.TOP_K_RETRIEVAL]
    candidates = [chunks[i] for i in candidate_idx]

    # ── Stage 2: cross-encoder rerank ──
    reranker = _get_reranker()
    pairs = [[query, chunk] for chunk in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    # Sort by reranker score descending, keep top-K
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, _ in ranked[: config.TOP_K]]
    top_scores = [float(s) for _, s in ranked[: config.TOP_K]]

    return top_chunks, top_scores
