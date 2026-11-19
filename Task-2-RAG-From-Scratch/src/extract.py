from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf

from src.config import EXTRACTION_STATS_PATH, PAGES_PATH, PDF_PATH, ensure_dirs


def normalize_whitespace(text: str) -> str:
    """Conservatively normalize PDF extraction artifacts without rewriting content.

    - Unify newlines
    - Join hyphenated line wraps (layout artifacts)
    - Collapse horizontal whitespace
    - Collapse 3+ blank lines
    - Strip each line
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(pdf_path: Path = PDF_PATH) -> tuple[list[dict], dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Required source PDF is missing: {pdf_path}. "
            "Do not substitute a different document."
        )

    doc = pymupdf.open(pdf_path)
    pages: list[dict] = []
    extracted_chars = 0
    usable_pages = 0

    for index, page in enumerate(doc):
        page_number = index + 1  # physical PDF page, 1-based
        raw = page.get_text() or ""
        extracted_chars += len(raw)
        normalized = normalize_whitespace(raw)
        is_usable = bool(normalized)
        if is_usable:
            usable_pages += 1
        pages.append(
            {
                "page_number": page_number,
                "text": normalized,
                "raw_char_count": len(raw),
                "normalized_char_count": len(normalized),
                "usable": is_usable,
            }
        )

    stats = {
        "pdf_path": str(pdf_path),
        "total_pages": len(doc),
        "usable_pages": usable_pages,
        "extracted_characters": extracted_chars,
        "normalized_characters": sum(p["normalized_char_count"] for p in pages),
        "empty_pages": [p["page_number"] for p in pages if not p["usable"]],
    }
    doc.close()
    return pages, stats


def save_pages(pages: list[dict], stats: dict) -> None:
    ensure_dirs()
    PAGES_PATH.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
    EXTRACTION_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_pages() -> list[dict]:
    if not PAGES_PATH.exists():
        raise FileNotFoundError("No extracted pages found. Run: python run.py ingest")
    return json.loads(PAGES_PATH.read_text(encoding="utf-8"))


def load_extraction_stats() -> dict:
    if not EXTRACTION_STATS_PATH.exists():
        raise FileNotFoundError("No extraction stats found. Run: python run.py ingest")
    return json.loads(EXTRACTION_STATS_PATH.read_text(encoding="utf-8"))


def page_text_by_number(pages: list[dict] | None = None) -> dict[int, str]:
    if pages is None:
        pages = load_pages()
    return {int(p["page_number"]): p["text"] for p in pages}


def quote_occurs_on_page(quote: str, page_text: str) -> bool:
    if not quote or not page_text:
        return False
    return compact_text(quote) in compact_text(page_text)
