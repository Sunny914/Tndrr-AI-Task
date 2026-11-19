from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunk import chunk_pages, load_chunks, save_chunks
from src.config import CHUNK_CONFIGS, GOLD_QUESTIONS_PATH, PDF_PATH, ensure_dirs
from src.embed import embed_or_load, load_embedding_model
from src.evaluate import evaluate_retrieval
from src.extract import extract_pages, save_pages
from src.generate import answer_question, ollama_available
from src.questions import (
    format_question_for_review,
    generate_candidates,
    load_gold_questions,
    set_verified,
    validate_gold_alignment,
)


def cmd_ingest(args: argparse.Namespace) -> int:
    ensure_dirs()
    print(f"Extracting {PDF_PATH} ...")
    pages, stats = extract_pages(PDF_PATH)
    save_pages(pages, stats)
    print(
        f"Pages: {stats['total_pages']} | usable: {stats['usable_pages']} | "
        f"extracted characters: {stats['extracted_characters']}"
    )
    if stats["total_pages"] < 30:
        print("ERROR: document has fewer than 30 pages.", file=sys.stderr)
        return 1

    print(f"Loading embedding model for tokenization and encoding ...")
    model = load_embedding_model()
    print(f"max_seq_length={model.max_seq_length}")

    for config_id, cfg in CHUNK_CONFIGS.items():
        print(f"Chunking config {config_id}: size={cfg['chunk_size']} overlap={cfg['overlap']}")
        chunks = chunk_pages(pages, config_id, model)
        save_chunks(chunks, config_id)
        embeddings = embed_or_load(config_id, chunks=chunks, model=model, force=args.force_embed)
        print(
            f"  chunks={len(chunks)} embeddings={embeddings.shape} "
            f"max_special_tokens={max(c['token_count_with_specials'] for c in chunks)}"
        )
    print("Ingest complete.")
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    rows = generate_candidates(preserve_verification=not args.force)
    errors = validate_gold_alignment(load_gold_questions(verified_only=False))
    if errors:
        print("Alignment errors:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    verified = sum(1 for row in rows if row["human_verified"] == "true")
    print(f"Wrote {len(rows)} candidate questions to {GOLD_QUESTIONS_PATH}")
    print(f"human_verified: {verified}/{len(rows)}")
    print("None of these are reported as evaluation-ready until you mark them verified.")
    print("Review with: python run.py review")
    print("Or edit the human_verified column in evaluation/gold_questions.csv")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    rows = load_gold_questions(verified_only=False)
    print("Human review of gold questions.")
    print("For each item, confirm that the evidence quote actually supports the question")
    print("and appears on the listed physical PDF page. Programmatic alignment is not")
    print("a substitute for human_verified=true.\n")
    if not sys.stdin.isatty() and not args.yes_all:
        print("Non-interactive session. Showing candidates only.")
        print("Mark verified later with: python run.py verify --ids Q01,Q02,...\n")
        for row in rows:
            print(format_question_for_review(row))
            print("-" * 72)
        return 0

    for row in rows:
        print(format_question_for_review(row))
        if args.yes_all:
            choice = "y"
        else:
            choice = input("Mark as human-verified? [y/N/q] ").strip().lower()
        if choice == "q":
            break
        if choice == "y":
            set_verified([row["id"]], True)
            print(f"  marked {row['id']} verified")
        print("-" * 72)
    verified = load_gold_questions(verified_only=True)
    print(f"Currently verified: {len(verified)}/{len(rows)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ids = [part.strip() for part in args.ids.split(",") if part.strip()]
    set_verified(ids, verified=not args.unverify)
    verified = load_gold_questions(verified_only=True)
    total = load_gold_questions(verified_only=False)
    print(f"Updated {ids}. Verified now: {len(verified)}/{len(total)}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_retrieval(verified_only=not args.allow_unverified)
    if result.get("status") == "blocked":
        print(result["reason"])
        return 2
    print(json.dumps({k: result[k] for k in result if k != "failure_cases"}, indent=2))
    print("\nFailure cases documented:", result.get("n_failure_cases_documented"))
    for case in result.get("failure_cases", []):
        print(f"- {case['kind']} {case['id']} config={case['config_id']} gold={case['gold_pages']} retrieved={case['retrieved_pages']}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    result = answer_question(args.question, config_id=args.config)
    print(f"Config: {result['config_id']}")
    print(f"Retrieved pages (retriever, not LLM): {result['retrieved_pages']}")
    for item in result["retrieved"]:
        preview = item["text"].replace("\n", " ")[:180]
        print(f"  #{item['rank']} page {item['page_number']} score={item['score']:.4f} {item['chunk_id']}")
        print(f"     {preview}")
    if result["answer"] is None:
        print(f"\nGeneration error: {result['error']}")
        return 3
    print("\nAnswer:\n")
    print(result["answer"])
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    import unittest

    from tests.test_pipeline import (
        ExtractionTests,
        IngestCacheTests,
        JsonCsvParseTests,
        MetricTests,
        QuoteAlignmentTests,
    )

    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for cls in (
        ExtractionTests,
        MetricTests,
        QuoteAlignmentTests,
        JsonCsvParseTests,
        IngestCacheTests,
    ):
        suite.addTests(loader.loadTestsFromTestCase(cls))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def cmd_status(args: argparse.Namespace) -> int:
    available, status = ollama_available()
    print(f"PDF exists: {PDF_PATH.exists()}")
    print(f"Ollama: {status}")
    print(f"Ollama available: {available}")
    for config_id in CHUNK_CONFIGS:
        try:
            chunks = load_chunks(config_id)
            print(f"Config {config_id}: {len(chunks)} chunks")
        except FileNotFoundError:
            print(f"Config {config_id}: not ingested")
    if GOLD_QUESTIONS_PATH.exists():
        total = load_gold_questions(verified_only=False)
        verified = load_gold_questions(verified_only=True)
        print(f"Gold questions: {len(verified)}/{len(total)} verified")
    else:
        print("Gold questions: not generated")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Framework-free RAG over NIST AI RMF 1.0"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Extract PDF, chunk, and embed both configs")
    ingest.add_argument("--force-embed", action="store_true", help="Recompute embeddings")
    ingest.set_defaults(func=cmd_ingest)

    candidates = sub.add_parser("candidates", help="Write 15 grounded evaluation-question candidates")
    candidates.add_argument("--force", action="store_true", help="Overwrite existing verification flags")
    candidates.set_defaults(func=cmd_candidates)

    review = sub.add_parser("review", help="Interactively mark gold questions as human-verified")
    review.add_argument("--yes-all", action="store_true", help="Mark every candidate verified (use only after reading them)")
    review.set_defaults(func=cmd_review)

    verify = sub.add_parser("verify", help="Mark selected question ids as human-verified")
    verify.add_argument("--ids", required=True, help="Comma-separated ids, e.g. Q01,Q02")
    verify.add_argument("--unverify", action="store_true")
    verify.set_defaults(func=cmd_verify)

    evaluate = sub.add_parser("evaluate", help="Measure retrieval for verified questions")
    evaluate.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Developer-only: evaluate candidates that are not human-verified. Do not report these as final scores.",
    )
    evaluate.set_defaults(func=cmd_evaluate)

    ask = sub.add_parser("ask", help="Retrieve and generate a grounded answer")
    ask.add_argument("question")
    ask.add_argument("--config", default="A", choices=list(CHUNK_CONFIGS))
    ask.set_defaults(func=cmd_ask)

    test = sub.add_parser("test", help="Run lightweight pipeline tests")
    test.set_defaults(func=cmd_test)

    status = sub.add_parser("status", help="Show ingest/Ollama/gold-question status")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
