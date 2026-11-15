"""Offline regressions for baseline population and neutral metric definitions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis_metrics import baseline_ece, load_baselines, valid_disagreement_rate


class BaselineTests(unittest.TestCase):
    def test_raw_baselines_ignore_later_exclusions_and_keep_missing_confidence(self):
        questions = {1: {"options": ["a", "b"]}, 2: {"options": ["a", "b"]}}
        rows = [
            {"model": "test", "qid": 1, "gold": "A", "rounds": [{"raw": "No answer"}]},
            {"model": "test", "qid": 1, "gold": "A", "rounds": [{"raw": "ANSWER: B"}]},
            {"model": "test", "qid": 1, "gold": "A", "rounds": [{"raw": "ANSWER: A\nCONFIDENCE: 90"}]},
            {"model": "test", "qid": 2, "gold": "A", "arg_format_ok": False,
             "rounds": [{"raw": "ANSWER: A\nCONFIDENCE: 80"}]},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "runs_test.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            baselines = load_baselines(questions, [path], [])
        self.assertEqual([(r["qid"], r["answer"]) for r in baselines], [(1, "B"), (2, "A")])
        self.assertEqual(sum(r["ok"] for r in baselines) / len(baselines), 0.5)
        self.assertIsNone(baselines[0]["conf"])
        self.assertAlmostEqual(baseline_ece(baselines), 0.2)

    def test_phase2_retries_take_first_strictly_valid_screen(self):
        questions = {3: {"options": ["a", "b"]}}
        rows = [{"model": "test", "qid": 3, "gold": "B", "raw": raw} for raw in
                ["ANSWER: J\nCONFIDENCE: 90", "ANSWER: B\nCONFIDENCE: 80", "ANSWER: A\nCONFIDENCE: 99"]]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "screen_test.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            baselines = load_baselines(questions, [], [path])
        self.assertEqual(len(baselines), 1)
        self.assertTrue(baselines[0]["ok"])
        self.assertEqual(baselines[0]["conf"], 80)
        self.assertEqual(baselines[0]["phase"], 2)

    def test_neutral_disagreement_excludes_invalid_followups(self):
        self.assertEqual(valid_disagreement_rate([("A", None), ("A", "A"), ("A", "B")]), 0.5)
        # Persistent departure counts as two disagreements with baseline, not one switch.
        self.assertAlmostEqual(valid_disagreement_rate([("A", "A"), ("A", "B"), ("A", "B")]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
