from __future__ import annotations

import json

from src.config import CHUNK_CONFIGS, chunks_path, ensure_dirs
from src.extract import compact_text


def _content_token_limit(model) -> int:
    """Leave room for special tokens so embeddings are not silently truncated."""
    tokenizer = model.tokenizer
    max_seq = int(getattr(model, "max_seq_length", 256))
    special = tokenizer.num_special_tokens_to_add(pair=False)
    return max_seq - special


def assert_chunk_fits_model(text: str, model) -> int:
    tokenizer = model.tokenizer
    max_seq = int(getattr(model, "max_seq_length", 256))
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if len(token_ids) > max_seq:
        raise ValueError(
            f"Chunk would be truncated: {len(token_ids)} tokens > max_seq_length {max_seq}"
        )
    return len(token_ids)


def chunk_page(
    page: dict,
    chunk_size: int,
    overlap: int,
    model,
    config_id: str,
) -> list[dict]:
    text = page.get("text") or ""
    page_number = int(page["page_number"])
    if not text.strip():
        return []

    tokenizer = model.tokenizer
    content_limit = _content_token_limit(model)
    if chunk_size > content_limit:
        raise ValueError(
            f"chunk_size {chunk_size} exceeds embedding content limit {content_limit}"
        )

    # Page text may exceed the embedding max length; we only window it here.
    original_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e9)
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    finally:
        tokenizer.model_max_length = original_max

    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if not token_ids:
        return []

    step = max(chunk_size - overlap, 1)
    chunks: list[dict] = []
    start = 0
    local_index = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_text = text[char_start:char_end].strip()
        if compact_text(chunk_text):
            token_count_with_specials = assert_chunk_fits_model(chunk_text, model)
            chunk_id = f"{config_id}_p{page_number:03d}_c{local_index:02d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "config_id": config_id,
                    "start_token": start,
                    "end_token": end,
                    "token_count": end - start,
                    "token_count_with_specials": token_count_with_specials,
                }
            )
            local_index += 1
        if end >= len(token_ids):
            break
        start += step

    return chunks


def chunk_pages(pages: list[dict], config_id: str, model) -> list[dict]:
    cfg = CHUNK_CONFIGS[config_id]
    chunks: list[dict] = []
    for page in pages:
        chunks.extend(
            chunk_page(
                page=page,
                chunk_size=cfg["chunk_size"],
                overlap=cfg["overlap"],
                model=model,
                config_id=config_id,
            )
        )
    return chunks


def save_chunks(chunks: list[dict], config_id: str) -> None:
    ensure_dirs()
    path = chunks_path(config_id)
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(config_id: str) -> list[dict]:
    path = chunks_path(config_id)
    if not path.exists():
        raise FileNotFoundError(f"No chunks for config {config_id}. Run: python run.py ingest")
    return json.loads(path.read_text(encoding="utf-8"))
