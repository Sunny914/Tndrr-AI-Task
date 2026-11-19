from __future__ import annotations

import csv
import re
from pathlib import Path

from src.config import GOLD_QUESTIONS_PATH, ensure_dirs
from src.extract import compact_text, load_pages, page_text_by_number, quote_occurs_on_page
from src.gold_specs import QUESTION_SPECS

FIELDNAMES = [
    "id",
    "question",
    "expected_answer",
    "gold_page",
    "gold_pages",
    "evidence_quote",
    "human_verified",
    "quote_found_on_gold_page",
    "notes",
]


def _normalize_for_search(text: str) -> str:
    text = compact_text(text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def _expand_quote(page_text: str, needle: str) -> str:
    """Return a sentence-bounded quote that starts at the supporting evidence."""
    compact = compact_text(page_text)
    haystack = _normalize_for_search(compact)
    needle_norm = _normalize_for_search(needle)
    # Replacements in _normalize_for_search are 1-to-1, so indexes match compact.
    idx = haystack.find(needle_norm)
    if idx < 0:
        loose_page = re.sub(r"[^a-z0-9]+", " ", haystack)
        loose_needle = re.sub(r"[^a-z0-9]+", " ", needle_norm).strip()
        if loose_needle not in loose_page:
            raise ValueError(f"Needle not found on page: {needle!r}")
        words = loose_needle.split()
        idx = haystack.find(words[0])
        last = haystack.rfind(words[-1])
        if idx < 0 or last < idx:
            raise ValueError(f"Needle words not recoverable on page: {needle!r}")
        end = last + len(words[-1])
    else:
        end = idx + len(needle_norm)
    period = compact.find(".", end)
    if period >= 0 and period - idx < 500:
        end = period + 1
    quote = compact[idx:end].strip(" ;:-")
    if not quote:
        raise ValueError(f"Empty recovered quote for needle: {needle!r}")
    return quote


def _load_existing_verification() -> dict[str, str]:
    if not GOLD_QUESTIONS_PATH.exists():
        return {}
    verified: dict[str, str] = {}
    with GOLD_QUESTIONS_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            verified[row["id"]] = row.get("human_verified", "false")
    return verified


def generate_candidates(preserve_verification: bool = True) -> list[dict]:
    pages = load_pages()
    by_page = page_text_by_number(pages)
    previous = _load_existing_verification() if preserve_verification else {}
    rows: list[dict] = []

    for spec in QUESTION_SPECS:
        gold_pages = [int(p) for p in spec["gold_pages"]]
        quotes = []
        found_all = True
        for page_number in gold_pages:
            if page_number not in by_page:
                raise ValueError(f"{spec['id']}: PDF has no page {page_number}")
            page_text = by_page[page_number]
            quote = _expand_quote(page_text, spec["needle"])
            if not quote_occurs_on_page(quote, page_text):
                found_all = False
            quotes.append(quote)
        evidence = " | ".join(quotes)
        human_verified = previous.get(spec["id"], "false")
        if str(human_verified).strip().lower() not in {"true", "false"}:
            human_verified = "false"
        rows.append(
            {
                "id": spec["id"],
                "question": spec["question"],
                "expected_answer": spec["expected_answer"],
                "gold_page": str(gold_pages[0]),
                "gold_pages": ";".join(str(p) for p in gold_pages),
                "evidence_quote": evidence,
                "human_verified": str(human_verified).lower(),
                "quote_found_on_gold_page": "true" if found_all else "false",
                "notes": spec["notes"],
            }
        )
        if not found_all:
            raise ValueError(f"{spec['id']} evidence quote did not align with gold page(s) {gold_pages}")

    ensure_dirs()
    write_gold_csv(rows, GOLD_QUESTIONS_PATH)
    return rows


def write_gold_csv(rows: list[dict], path: Path = GOLD_QUESTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_gold_questions(verified_only: bool = False) -> list[dict]:
    if not GOLD_QUESTIONS_PATH.exists():
        raise FileNotFoundError("Gold questions CSV is missing. Run: python run.py candidates")
    rows: list[dict] = []
    with GOLD_QUESTIONS_PATH.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            raw["gold_pages_list"] = [
                int(part) for part in raw["gold_pages"].split(";") if part.strip()
            ]
            raw["human_verified_bool"] = str(raw.get("human_verified", "")).strip().lower() == "true"
            if verified_only and not raw["human_verified_bool"]:
                continue
            rows.append(raw)
    return rows


def validate_gold_alignment(rows: list[dict] | None = None) -> list[str]:
    if rows is None:
        rows = load_gold_questions(verified_only=False)
    by_page = page_text_by_number()
    errors: list[str] = []
    for row in rows:
        for page_number in row["gold_pages_list"]:
            page_text = by_page.get(page_number, "")
            if not quote_occurs_on_page(row["evidence_quote"].split(" | ")[0], page_text):
                # Multi-page rows join quotes with " | "; validate each piece against any gold page.
                matched = False
                for piece in row["evidence_quote"].split(" | "):
                    if quote_occurs_on_page(piece, page_text):
                        matched = True
                        break
                if not matched:
                    errors.append(
                        f"{row['id']}: evidence quote not found on physical PDF page {page_number}"
                    )
    return errors


def set_verified(question_ids: list[str], verified: bool) -> None:
    rows = load_gold_questions(verified_only=False)
    wanted = {qid.strip().upper() for qid in question_ids}
    found = set()
    for row in rows:
        if row["id"] in wanted:
            row["human_verified"] = "true" if verified else "false"
            found.add(row["id"])
    missing = wanted - found
    if missing:
        raise ValueError(f"Unknown question ids: {sorted(missing)}")
    # Drop helper keys before write.
    cleaned = [{k: row[k] for k in FIELDNAMES} for row in rows]
    write_gold_csv(cleaned)


def format_question_for_review(row: dict) -> str:
    return (
        f"{row['id']}\n"
        f"Question: {row['question']}\n"
        f"Expected answer: {row['expected_answer']}\n"
        f"Gold PDF page(s): {row['gold_pages']}\n"
        f"Evidence quote: {row['evidence_quote']}\n"
        f"Programmatic page/quote alignment: {row['quote_found_on_gold_page']}\n"
        f"human_verified: {row['human_verified']}\n"
        f"Notes: {row['notes']}"
    )
