"""
tests/test_evaluation.py — tests for the evaluation harness and dataset.
"""

from pathlib import Path
import pytest

from src.chunker import load_policy
from src.retriever import BM25Retriever
from src.evaluator import EVALUATION_DATASET, evaluate_question, run_all_evaluations
from src.pipeline import GroundedAnswerPipeline

MANUAL_PATH = Path(__file__).parent.parent / "data" / "policy-manual.md"


@pytest.fixture(scope="module")
def pipeline():
    return GroundedAnswerPipeline.build(MANUAL_PATH)


def test_evaluation_dataset_has_at_least_10_questions():
    assert len(EVALUATION_DATASET) >= 10


def test_evaluation_dataset_covers_required_categories():
    categories = {case["category"] for case in EVALUATION_DATASET}
    statuses = {case["expected_status"] for case in EVALUATION_DATASET}

    assert "answered" in statuses
    assert "refusal" in statuses
    assert "contradiction" in statuses
    assert "gap" in statuses


def test_contradiction_question_retrieves_both_conflicting_clauses(pipeline):
    """The contradiction question must retrieve both §4.3.2 and §9.1.4."""
    case = next(c for c in EVALUATION_DATASET if c["expected_status"] == "contradiction")
    evidence = pipeline._retriever.retrieve(case["question"], top_k=5)
    evidence_ids = [c["id"] for c in evidence]

    assert "§4.3.2" in evidence_ids
    assert "§9.1.4" in evidence_ids


def test_student_gap_question_retrieves_award_clause(pipeline):
    """The student gap question must retrieve §7.1.3."""
    case = next(c for c in EVALUATION_DATASET if c["expected_status"] == "gap")
    evidence = pipeline._retriever.retrieve(case["question"], top_k=5)
    evidence_ids = [c["id"] for c in evidence]

    assert "§7.1.3" in evidence_ids


def test_dry_run_evaluation_passes_all_cases(pipeline):
    """Dry-run evaluation checks evidence retrieval across all cases."""
    results = run_all_evaluations(pipeline, dry_run=True)
    assert len(results) == len(EVALUATION_DATASET)
    assert all(r["passed"] for r in results)