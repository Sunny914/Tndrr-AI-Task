from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHUNK_CONFIGS, PDF_PATH, chunks_path, embeddings_path
from src.evaluate import hit_at_k, quote_hit, reciprocal_rank, summarize_rows
from src.extract import extract_pages, quote_occurs_on_page
from src.retrieve import cosine_scores, ranked_indices


class ExtractionTests(unittest.TestCase):
    def test_pdf_exists_and_has_enough_pages(self):
        self.assertTrue(PDF_PATH.exists(), f"Missing PDF: {PDF_PATH}")
        pages, stats = extract_pages(PDF_PATH)
        self.assertGreaterEqual(stats["total_pages"], 30)
        self.assertGreaterEqual(stats["usable_pages"], 30)
        self.assertGreater(stats["extracted_characters"], 0)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("NIST AI 100-1", pages[0]["text"])
        self.assertEqual(pages[-1]["page_number"], stats["total_pages"])


class MetricTests(unittest.TestCase):
    def test_hit_mrr_and_quote_on_synthetic_ranks(self):
        gold = [5]
        self.assertEqual(hit_at_k(gold, [5, 8, 9, 1, 2], 1), 1)
        self.assertEqual(hit_at_k(gold, [5, 8, 9, 1, 2], 5), 1)
        self.assertEqual(reciprocal_rank(gold, [5, 8, 9, 1, 2], 5), 1.0)

        self.assertEqual(hit_at_k([7], [3, 7, 1, 2, 4], 1), 0)
        self.assertEqual(hit_at_k([7], [3, 7, 1, 2, 4], 5), 1)
        self.assertEqual(reciprocal_rank([7], [3, 7, 1, 2, 4], 5), 0.5)

        self.assertEqual(hit_at_k([10], [1, 2, 3, 4, 5], 1), 0)
        self.assertEqual(hit_at_k([10], [1, 2, 3, 4, 5], 5), 0)
        self.assertEqual(reciprocal_rank([10], [1, 2, 3, 4, 5], 5), 0.0)

        self.assertEqual(quote_hit("Residual risk remaining after risk treatment", [
            "unrelated chunk",
            "Residual risk  remaining   after risk treatment (ISO).",
        ]), 1)
        self.assertEqual(quote_hit("Residual risk remaining after risk treatment", [
            "risk remaining in the abstract",
        ]), 0)

        summary = summarize_rows(
            [
                {"hit_at_1": 1, "hit_at_5": 1, "rr": 1.0, "quote_hit_at_5": 1},
                {"hit_at_1": 0, "hit_at_5": 1, "rr": 0.5, "quote_hit_at_5": 0},
            ]
        )
        self.assertEqual(summary["n_questions"], 2)
        self.assertEqual(summary["hit_at_1"], 0.5)
        self.assertEqual(summary["hit_at_5"], 1.0)
        self.assertEqual(summary["mrr"], 0.75)

    def test_descending_similarity_order(self):
        import numpy as np

        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.7071, 0.7071, 0.0],
            ],
            dtype=np.float32,
        )
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        scores = cosine_scores(query, embeddings)
        order = ranked_indices(scores)
        self.assertEqual(list(order), [0, 2, 1])
        self.assertTrue(scores[order][0] >= scores[order][1] >= scores[order][2])


class IngestCacheTests(unittest.TestCase):
    def test_cached_chunks_and_embeddings_if_present(self):
        for config_id, cfg in CHUNK_CONFIGS.items():
            chunk_file = chunks_path(config_id)
            emb_file = embeddings_path(config_id)
            if not chunk_file.exists() or not emb_file.exists():
                self.skipTest("Run python run.py ingest before cache assertions.")
            chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
            self.assertGreater(len(chunks), 0)
            import numpy as np

            embeddings = np.load(emb_file)
            self.assertGreater(embeddings.size, 0)
            self.assertEqual(embeddings.shape[0], len(chunks))
            self.assertTrue(all(c["chunk_size"] == cfg["chunk_size"] for c in chunks))
            self.assertTrue(all(c["page_number"] >= 1 for c in chunks))
            self.assertTrue(all(c["end_token"] > c["start_token"] for c in chunks))
            self.assertTrue(all(c["token_count_with_specials"] <= 256 for c in chunks))


class QuoteAlignmentTests(unittest.TestCase):
    def test_known_cover_quote(self):
        pages, _ = extract_pages(PDF_PATH)
        self.assertTrue(quote_occurs_on_page("Artificial Intelligence Risk Management", pages[0]["text"]))
        self.assertFalse(quote_occurs_on_page("this quote is not in the document", pages[0]["text"]))

    def test_gold_csv_quotes_align_if_present(self):
        from src.config import GOLD_QUESTIONS_PATH
        from src.questions import load_gold_questions, validate_gold_alignment

        if not GOLD_QUESTIONS_PATH.exists():
            self.skipTest("Run python run.py candidates first.")
        rows = load_gold_questions(verified_only=False)
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(row["human_verified"] in {"true", "false"} for row in rows))
        errors = validate_gold_alignment(rows)
        self.assertEqual(errors, [])


class JsonCsvParseTests(unittest.TestCase):
    def test_roundtrip_temp_outputs(self):
        payload = {"hit_at_5": 0.8, "n": 5}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation_summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["n"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
