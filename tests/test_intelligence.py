from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cuet_bst.intelligence import (  # noqa: E402
    apply_manual_labels,
    classify_question_pattern,
    normalize_question_identity,
    source_tier,
    source_weight,
    year_weight,
)
from cuet_bst.validation import apply_taxonomy_validation  # noqa: E402


class IntelligenceTests(TestCase):
    def test_normalization_removes_punctuation_and_noise(self) -> None:
        row = {
            "question_text": "Which option is Correct?  Business Studies!!",
            "option_a": "A. Planning",
            "option_b": "B) Controlling",
            "option_c": "",
            "option_d": "",
        }
        normalized = normalize_question_identity(row)
        self.assertNotIn("correct", normalized)
        self.assertIn("planning", normalized)
        self.assertIn("controlling", normalized)

    def test_source_tiers_and_weights(self) -> None:
        self.assertEqual(source_tier("NTA paper", "https://nta.ac.in/Download/ExamPaper/file.pdf"), "official_nta")
        self.assertEqual(source_tier("official_manual", "manual-official://paper.pdf"), "official_manual")
        self.assertEqual(source_weight("AfterBoards", "https://afterboards.in/past-year-questions"), 0.65)
        self.assertEqual(source_weight("official_manual", "manual-official://paper.pdf"), 1.0)
        self.assertEqual(source_weight("Reddit", "https://reddit.com/r/CUETards"), 0.15)

    def test_year_weight(self) -> None:
        self.assertEqual(year_weight("2025"), 1.0)
        self.assertEqual(year_weight("2022.0"), 0.45)
        self.assertEqual(year_weight(""), 0.35)

    def test_question_pattern_classifier(self) -> None:
        self.assertEqual(classify_question_pattern("Match List-I with List-II"), "match-the-following")
        self.assertEqual(classify_question_pattern("Assertion (A) and Reason (R)"), "assertion-reason")
        self.assertEqual(classify_question_pattern("Arrange the steps in correct sequence"), "chronology/process order")

    def test_manual_label_override_precedence(self) -> None:
        labels = pd.DataFrame(
            [
                {
                    "canonical_question_id": "cq_test",
                    "chapter": "Planning",
                    "subtopic": "types of plans",
                    "question_pattern": "definition-based",
                    "difficulty_estimate": "easy",
                    "ncert_heading": "Types Of Plans",
                    "micro_concept": "budget",
                }
            ]
        )
        df = pd.DataFrame(
            [
                {
                    "canonical_question_id": "cq_test",
                    "chapter": "Old",
                    "subtopic": "Old",
                    "question_pattern": "Old",
                    "difficulty_estimate": "hard",
                    "ncert_heading": "",
                    "micro_concept": "",
                    "needs_review": "true",
                    "review_reason": "",
                }
            ]
        )
        out = apply_manual_labels(df, labels)
        self.assertEqual(out.iloc[0]["chapter"], "Planning")
        self.assertEqual(out.iloc[0]["micro_concept"], "budget")
        self.assertEqual(out.iloc[0]["needs_review"], "false")

    def test_taxonomy_mismatch_detection(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "canonical_question_id": "cq_bad",
                    "chapter": "Marketing",
                    "subtopic": "financing decision",
                    "micro_concept": "financial risk",
                    "question_pattern": "definition-based",
                    "final_confidence": "0.9",
                    "needs_review": "false",
                    "review_reason": "",
                    "question_text": "Debt and equity question",
                    "source_url": "",
                }
            ]
        )
        out = apply_taxonomy_validation(df)
        self.assertEqual(str(out.iloc[0]["taxonomy_mismatch"]).lower(), "true")
        self.assertEqual(out.iloc[0]["needs_review"], "true")
