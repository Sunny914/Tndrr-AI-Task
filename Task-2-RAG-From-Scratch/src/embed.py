from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunk import load_chunks
from src.config import EMBEDDING_MODEL_NAME, embeddings_path, ensure_dirs


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _texts_match_cache(chunks: list[dict], config_id: str) -> bool:
    chunk_file = embeddings_path(config_id).with_suffix(".meta.json")
    npy_path = embeddings_path(config_id)
    if not npy_path.exists() or not chunk_file.exists():
        return False
    meta = json.loads(chunk_file.read_text(encoding="utf-8"))
    cached_ids = meta.get("chunk_ids", [])
    cached_texts = meta.get("chunk_texts", [])
    cached_model = meta.get("model_name")
    current_ids = [c["chunk_id"] for c in chunks]
    current_texts = [c["text"] for c in chunks]
    return (
        cached_model == EMBEDDING_MODEL_NAME
        and cached_ids == current_ids
        and cached_texts == current_texts
    )


def embed_chunks(chunks: list[dict], model: SentenceTransformer | None = None) -> np.ndarray:
    if not chunks:
        raise ValueError("Cannot embed an empty chunk list.")
    if model is None:
        model = load_embedding_model()

    texts = [c["text"] for c in chunks]
    tokenizer = model.tokenizer
    max_seq = int(getattr(model, "max_seq_length", 256))
    for text in texts:
        n_tokens = len(tokenizer.encode(text, add_special_tokens=True))
        if n_tokens > max_seq:
            raise ValueError(
                f"Refusing to embed a truncated sequence: {n_tokens} > {max_seq}"
            )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.size == 0 or embeddings.ndim != 2:
        raise ValueError("Embedding array is empty or has unexpected shape.")
    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")
    return embeddings


def save_embeddings(embeddings: np.ndarray, chunks: list[dict], config_id: str) -> None:
    ensure_dirs()
    npy_path = embeddings_path(config_id)
    meta_path = npy_path.with_suffix(".meta.json")
    np.save(npy_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "model_name": EMBEDDING_MODEL_NAME,
                "config_id": config_id,
                "shape": list(embeddings.shape),
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "chunk_texts": [c["text"] for c in chunks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_embeddings(config_id: str) -> np.ndarray:
    npy_path = embeddings_path(config_id)
    if not npy_path.exists():
        raise FileNotFoundError(
            f"No embeddings for config {config_id}. Run: python run.py ingest"
        )
    embeddings = np.load(npy_path)
    if embeddings.size == 0:
        raise ValueError(f"Cached embeddings for {config_id} are empty.")
    return embeddings


def embed_or_load(
    config_id: str,
    chunks: list[dict] | None = None,
    model: SentenceTransformer | None = None,
    force: bool = False,
) -> np.ndarray:
    if chunks is None:
        chunks = load_chunks(config_id)
    if not force and _texts_match_cache(chunks, config_id):
        return load_embeddings(config_id)
    embeddings = embed_chunks(chunks, model=model)
    save_embeddings(embeddings, chunks, config_id)
    return embeddings


def embed_query(question: str, model: SentenceTransformer | None = None) -> np.ndarray:
    if model is None:
        model = load_embedding_model()
    vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vector = np.asarray(vector, dtype=np.float32)[0]
    if vector.size == 0:
        raise ValueError("Query embedding is empty.")
    return vector
