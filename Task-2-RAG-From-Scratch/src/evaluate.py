from __future__ import annotations

import csv
import json
from statistics import mean

from src.config import CHUNK_CONFIGS, RESULTS_DIR, TOP_K, ensure_dirs
from src.extract import compact_text
from src.questions import load_gold_questions, validate_gold_alignment
from src.retrieve import retrieve

METRIC_DEFINITIONS = {
    "hit_at_1": (
        "Page-based supporting-evidence Hit@1: 1 if the rank-1 retrieved chunk's "
        "physical PDF page is one of the gold supporting pages, else 0. Reported as the "
        "mean over evaluated questions."
    ),
    "hit_at_5": (
        "Page-based supporting-evidence Hit@5: 1 if at least one of the top-5 retrieved "
        "chunks belongs to a verified supporting page, else 0. Reported as the mean over "
        "evaluated questions."
    ),
    "mrr": (
        "Mean Reciprocal Rank: for each question, 1/rank of the first top-5 chunk whose "
        "page is a gold supporting page, or 0 if none of the top-5 pages match. Mean over "
        "evaluated questions. Rank is 1-based."
    ),
    "quote_hit_at_5": (
        "Exact supporting-quote retrieval Hit@5: 1 if the verified evidence quote occurs "
        "(whitespace-normalized) in at least one of the top-5 retrieved chunk texts, else 0. "
        "This is stricter than page matching."
    ),
    "page_match_limitation": (
        "Page-level matching is a proxy for evidence relevance. A chunk can come from the "
        "correct PDF page yet miss the supporting sentence, so Hit@1/Hit@5 can overestimate "
        "true evidence retrieval. Quote Hit@5 is reported to expose that gap."
    ),
}


def parse_gold_pages(value) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(part) for part in str(value).split(";") if part.strip()]


def hit_at_k(gold_pages: list[int], retrieved_pages: list[int], k: int) -> int:
    return int(any(page in gold_pages for page in retrieved_pages[:k]))


def reciprocal_rank(gold_pages: list[int], retrieved_pages: list[int], k: int) -> float:
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in gold_pages:
            return 1.0 / rank
    return 0.0


def quote_hit(evidence_quote: str, retrieved_texts: list[str]) -> int:
    needle = compact_text(evidence_quote)
    if not needle:
        return 0
    return int(any(needle in compact_text(text) for text in retrieved_texts))


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_questions": 0,
            "hit_at_1": None,
            "hit_at_5": None,
            "mrr": None,
            "quote_hit_at_5": None,
        }
    return {
        "n_questions": len(rows),
        "hit_at_1": mean(row["hit_at_1"] for row in rows),
        "hit_at_5": mean(row["hit_at_5"] for row in rows),
        "mrr": mean(row["rr"] for row in rows),
        "quote_hit_at_5": mean(row["quote_hit_at_5"] for row in rows),
    }


def _failure_record(row: dict, kind: str) -> dict:
    return {
        "kind": kind,
        "id": row["id"],
        "question": row["question"],
        "config_id": row["config_id"],
        "gold_pages": row["gold_pages"],
        "retrieved_pages": row["retrieved_pages"],
        "retrieved_chunk_ids": row["retrieved_chunk_ids"],
        "hit_at_1": row["hit_at_1"],
        "hit_at_5": row["hit_at_5"],
        "quote_hit_at_5": row["quote_hit_at_5"],
        "evidence_quote": row["evidence_quote"],
        "why_retrieval_may_have_failed": (
            "The dense retriever ranked other pages above the verified supporting page. "
            "Likely causes include vocabulary mismatch, competing nearby sections that "
            "share framework terms, or chunk boundaries that dilute the supporting sentence."
        ),
        "chunk_size_observation": (
            "Compare the same question under configs A (120/20) and B (220/40) before "
            "concluding that smaller or larger chunks help."
        ),
        "possible_improvement": (
            "Hybrid BM25 + dense retrieval, reciprocal rank fusion, and cross-encoder "
            "reranking over a larger candidate pool."
        ),
    }


def select_failure_cases(detail_rows: list[dict], limit: int = 2) -> list[dict]:
    """Prefer unique Hit@5 misses; if fewer than two questions, use Hit@1 or quote misses."""
    selected: list[dict] = []
    seen_ids: set[str] = set()

    def add(row: dict, kind: str) -> None:
        if row["id"] in seen_ids or len(selected) >= limit:
            return
        selected.append(_failure_record(row, kind))
        seen_ids.add(row["id"])

    for row in detail_rows:
        if row["hit_at_5"] == 0:
            add(row, "hit_at_5_miss")
    # Prefer Hit@1 misses that also missed the evidence quote (page-level Hit@5 can overestimate).
    for row in detail_rows:
        if row["hit_at_1"] == 0 and row["quote_hit_at_5"] == 0:
            add(row, "hit_at_1_and_quote_miss")
    for row in detail_rows:
        if row["hit_at_1"] == 0:
            add(row, "hit_at_1_miss")
    for row in detail_rows:
        if row["quote_hit_at_5"] == 0:
            add(row, "quote_miss")
    return selected


def evaluate_retrieval(verified_only: bool = True, top_k: int = TOP_K) -> dict:
    questions = load_gold_questions(verified_only=verified_only)
    if verified_only and not questions:
        return {
            "status": "blocked",
            "reason": (
                "No human-verified gold questions. Review evaluation/gold_questions.csv "
                "and run `python run.py review` or `python run.py verify --ids Q01,Q02,...` "
                "before reporting retrieval accuracy."
            ),
            "n_verified": 0,
        }

    alignment_errors = validate_gold_alignment(questions)
    if alignment_errors:
        raise ValueError("Gold evidence/page alignment failed:\n" + "\n".join(alignment_errors))

    ensure_dirs()
    detail_rows: list[dict] = []
    comparison_rows: list[dict] = []

    for config_id, cfg in CHUNK_CONFIGS.items():
        config_details = []
        for question in questions:
            retrieved = retrieve(question["question"], config_id=config_id, top_k=top_k)
            retrieved_pages = [item["page_number"] for item in retrieved]
            retrieved_texts = [item["text"] for item in retrieved]
            gold_pages = question["gold_pages_list"]
            row = {
                "id": question["id"],
                "question": question["question"],
                "config_id": config_id,
                "chunk_size": cfg["chunk_size"],
                "overlap": cfg["overlap"],
                "gold_pages": ";".join(str(p) for p in gold_pages),
                "retrieved_pages": ";".join(str(p) for p in retrieved_pages),
                "retrieved_chunk_ids": ";".join(item["chunk_id"] for item in retrieved),
                "retrieved_scores": ";".join(f"{item['score']:.6f}" for item in retrieved),
                "hit_at_1": hit_at_k(gold_pages, retrieved_pages, 1),
                "hit_at_5": hit_at_k(gold_pages, retrieved_pages, 5),
                "rr": reciprocal_rank(gold_pages, retrieved_pages, 5),
                "quote_hit_at_5": quote_hit(question["evidence_quote"], retrieved_texts),
                "evidence_quote": question["evidence_quote"],
                "human_verified": question["human_verified"],
            }
            config_details.append(row)
            detail_rows.append(row)

        summary = summarize_rows(config_details)
        comparison_rows.append(
            {
                "config_id": config_id,
                "chunk_size": cfg["chunk_size"],
                "overlap": cfg["overlap"],
                "n_questions": summary["n_questions"],
                "hit_at_1": summary["hit_at_1"],
                "hit_at_5": summary["hit_at_5"],
                "mrr": summary["mrr"],
                "quote_hit_at_5": summary["quote_hit_at_5"],
            }
        )

    comparison_path = RESULTS_DIR / "chunk_comparison.csv"
    retrieval_path = RESULTS_DIR / "retrieval_results.csv"
    summary_path = RESULTS_DIR / "evaluation_summary.json"

    with retrieval_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()) if detail_rows else ["id"])
        writer.writeheader()
        writer.writerows(detail_rows)

    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_id",
                "chunk_size",
                "overlap",
                "n_questions",
                "hit_at_1",
                "hit_at_5",
                "mrr",
                "quote_hit_at_5",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    failures = select_failure_cases(detail_rows)
    a_summary = next(row for row in comparison_rows if row["config_id"] == "A")
    b_summary = next(row for row in comparison_rows if row["config_id"] == "B")
    if a_summary["hit_at_5"] > b_summary["hit_at_5"]:
        comparison_note = "Config A had higher page-based Hit@5 on this verified set."
    elif b_summary["hit_at_5"] > a_summary["hit_at_5"]:
        comparison_note = "Config B had higher page-based Hit@5 on this verified set."
    else:
        comparison_note = "Configs A and B had the same page-based Hit@5 on this verified set."

    payload = {
        "status": "measured",
        "n_verified_questions": len(questions),
        "top_k": top_k,
        "metrics": METRIC_DEFINITIONS,
        "chunk_comparison": comparison_rows,
        "comparison_note": comparison_note,
        "failure_cases": failures,
        "n_failure_cases_documented": len(failures),
        "n_unique_hit_at_5_miss_questions": len(
            {row["id"] for row in detail_rows if row["hit_at_5"] == 0}
        ),
        "failure_case_note": (
            "Fewer than two unique Hit@5-miss questions were available; a Hit@1 and/or "
            "quote-level miss was documented as the second case. No failed queries were invented."
            if len({row["id"] for row in detail_rows if row["hit_at_5"] == 0}) < 2
            else "Documented from actual Hit@5 misses."
        ),
        "bm25_and_reranking": {
            "bm25": (
                "BM25 would score lexical overlap between the question and chunk text. "
                "It helps when the gold sentence uses the same distinctive terms as the "
                "question (for example TEVV, residual risk, GOVERN) that dense MiniLM "
                "embeddings may under-weight."
            ),
            "hybrid": (
                "Hybrid search keeps both BM25 and dense candidate lists, covering exact "
                "term matches and paraphrases."
            ),
            "rrf": (
                "Reciprocal Rank Fusion combines the two ranked lists without needing "
                "score calibration, which is useful when BM25 scores and cosine scores "
                "are on different scales."
            ),
            "cross_encoder_reranking": (
                "A cross-encoder would jointly encode question and each top-N chunk, "
                "which is slower but typically better at rejecting same-page or nearby "
                "chunks that share topic words but not the supporting evidence."
            ),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["paths"] = {
        "chunk_comparison": str(comparison_path),
        "retrieval_results": str(retrieval_path),
        "evaluation_summary": str(summary_path),
    }
    return payload
