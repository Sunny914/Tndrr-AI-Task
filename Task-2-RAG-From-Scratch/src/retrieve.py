from __future__ import annotations

import numpy as np

from src.chunk import load_chunks
from src.config import TOP_K
from src.embed import embed_or_load, embed_query, load_embedding_model


def cosine_scores(query_vector: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity via matrix multiplication on L2-normalized vectors."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Chunk embeddings must be a 2-D array.")
    scores = matrix @ query
    return scores.astype(np.float32)


def ranked_indices(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    if len(order) > 1 and np.any(scores[order][1:] > scores[order][:-1] + 1e-12):
        raise RuntimeError("Retrieval ranking is not in descending similarity order.")
    return order


def retrieve(
    question: str,
    config_id: str,
    top_k: int = TOP_K,
    model=None,
) -> list[dict]:
    chunks = load_chunks(config_id)
    embeddings = embed_or_load(config_id, chunks=chunks, model=model)
    if model is None:
        model = load_embedding_model()
    query_vector = embed_query(question, model=model)
    scores = cosine_scores(query_vector, embeddings)
    order = ranked_indices(scores)
    k = min(top_k, len(order))
    results = []
    for rank, idx in enumerate(order[:k], start=1):
        chunk = chunks[int(idx)]
        results.append(
            {
                "rank": rank,
                "score": float(scores[int(idx)]),
                "chunk_id": chunk["chunk_id"],
                "page_number": int(chunk["page_number"]),
                "text": chunk["text"],
                "config_id": config_id,
            }
        )
    return results
