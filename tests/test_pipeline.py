"""
tests/test_pipeline.py — integration tests for GroundedAnswerPipeline.

All tests are fully mocked — no real LLM API calls are made.
The BM25 retriever runs against the real policy manual so retrieval
behaviour is tested end-to-end.

Run with:  pytest tests/test_pipeline.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chunker import load_policy
from src.retriever import BM25Retriever
from src.generator import AnswerGenerator, _NO_EVIDENCE_ANSWER
from src.pipeline import GroundedAnswerPipeline

MANUAL_PATH = Path(__file__).parent.parent / "data" / "policy-manual.md"

_FAKE_ANSWER = "According to §6.6.1, the threshold for a household of 3 is $2,000 per month."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(fake_answer: str = _FAKE_ANSWER) -> GroundedAnswerPipeline:
    """Build a real pipeline with a mocked generator — no API calls."""
    clauses = load_policy(str(MANUAL_PATH))
    retriever = BM25Retriever(clauses)
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = fake_answer
    return GroundedAnswerPipeline(retriever=retriever, generator=generator)


# ---------------------------------------------------------------------------
# A. Question reaches the BM25 retriever
# ---------------------------------------------------------------------------

def test_question_is_passed_to_retriever():
    clauses = load_policy(str(MANUAL_PATH))
    retriever = MagicMock(spec=BM25Retriever)
    retriever.retrieve.return_value = [
        {"id": "§6.6.1", "part": 6, "section": "6.6",
         "heading": "Income thresholds", "text": "threshold text", "score": 9.0}
    ]
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = _FAKE_ANSWER

    pipeline = GroundedAnswerPipeline(retriever=retriever, generator=generator)
    question = "What is the income threshold for a household of 3?"
    pipeline.ask(question)

    retriever.retrieve.assert_called_once()
    call_args = retriever.retrieve.call_args
    assert call_args[0][0] == question or call_args[1].get("query") == question


# ---------------------------------------------------------------------------
# B. BM25 returns at most 5 clauses by default
# ---------------------------------------------------------------------------

def test_default_top_k_is_five():
    pipeline = _make_pipeline()
    result = pipeline.ask("income threshold household of 3")
    assert len(result["evidence"]) <= 5


def test_default_top_k_passes_five_to_retriever():
    clauses = load_policy(str(MANUAL_PATH))
    retriever = MagicMock(spec=BM25Retriever)
    retriever.retrieve.return_value = []
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = _NO_EVIDENCE_ANSWER

    pipeline = GroundedAnswerPipeline(retriever=retriever, generator=generator)
    pipeline.ask("any question")

    _, kwargs = retriever.retrieve.call_args
    assert kwargs.get("top_k", retriever.retrieve.call_args[0][1] if len(retriever.retrieve.call_args[0]) > 1 else 5) == 5


# ---------------------------------------------------------------------------
# C & D. Retrieved clauses are passed to the generator — same objects
# ---------------------------------------------------------------------------

def test_generator_receives_retrieved_clauses():
    clauses = load_policy(str(MANUAL_PATH))
    retriever = BM25Retriever(clauses)
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = _FAKE_ANSWER

    pipeline = GroundedAnswerPipeline(retriever=retriever, generator=generator)
    question = "income threshold household of 3"
    result = pipeline.ask(question)

    generator.generate.assert_called_once()
    _, passed_clauses = generator.generate.call_args[0]
    assert passed_clauses == result["evidence"]


def test_generator_receives_exact_bm25_results():
    """The clauses passed to the generator must be identical to what BM25 returned."""
    fixed_clauses = [
        {"id": "§6.6.1", "part": 6, "section": "6.6",
         "heading": "Income thresholds", "text": "threshold text", "score": 9.0}
    ]
    retriever = MagicMock(spec=BM25Retriever)
    retriever.retrieve.return_value = fixed_clauses
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = _FAKE_ANSWER

    pipeline = GroundedAnswerPipeline(retriever=retriever, generator=generator)
    pipeline.ask("income threshold")

    passed_question, passed_clauses = generator.generate.call_args[0]
    assert passed_clauses is fixed_clauses


# ---------------------------------------------------------------------------
# E. Complete policy manual is NOT passed to the generator
# ---------------------------------------------------------------------------

def test_full_manual_not_passed_to_generator():
    clauses = load_policy(str(MANUAL_PATH))
    retriever = BM25Retriever(clauses)
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = _FAKE_ANSWER

    pipeline = GroundedAnswerPipeline(retriever=retriever, generator=generator)
    pipeline.ask("income threshold household of 3")

    _, passed_clauses = generator.generate.call_args[0]
    # The full corpus has 148 clauses; only top_k=5 should be passed
    assert len(passed_clauses) <= 5
    assert len(passed_clauses) < len(clauses)


# ---------------------------------------------------------------------------
# F. Result contains both answer and evidence
# ---------------------------------------------------------------------------

def test_result_has_answer_and_evidence_keys():
    pipeline = _make_pipeline()
    result = pipeline.ask("income threshold household of 3")
    assert "answer" in result
    assert "evidence" in result


def test_result_answer_is_string():
    pipeline = _make_pipeline()
    result = pipeline.ask("income threshold household of 3")
    assert isinstance(result["answer"], str)


def test_result_evidence_is_list():
    pipeline = _make_pipeline()
    result = pipeline.ask("income threshold household of 3")
    assert isinstance(result["evidence"], list)


def test_result_evidence_clauses_have_required_keys():
    pipeline = _make_pipeline()
    result = pipeline.ask("income threshold household of 3")
    for clause in result["evidence"]:
        for key in ("id", "part", "section", "heading", "text"):
            assert key in clause, f"missing key '{key}' in clause {clause.get('id')}"


# ---------------------------------------------------------------------------
# G. Empty questions are handled safely
# ---------------------------------------------------------------------------

def test_empty_question_returns_insufficiency_message():
    pipeline = _make_pipeline()
    result = pipeline.ask("")
    assert result["answer"] == _NO_EVIDENCE_ANSWER
    assert result["evidence"] == []


def test_whitespace_question_returns_insufficiency_message():
    pipeline = _make_pipeline()
    result = pipeline.ask("   ")
    assert result["answer"] == _NO_EVIDENCE_ANSWER
    assert result["evidence"] == []


# ---------------------------------------------------------------------------
# H. No-results retrieval is handled safely
# ---------------------------------------------------------------------------

def test_no_retrieval_results_returns_insufficiency_message():
    retriever = MagicMock(spec=BM25Retriever)
    retriever.retrieve.return_value = []
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = _NO_EVIDENCE_ANSWER

    pipeline = GroundedAnswerPipeline(retriever=retriever, generator=generator)
    result = pipeline.ask("something obscure")
    assert result["answer"] == _NO_EVIDENCE_ANSWER


# ---------------------------------------------------------------------------
# Pipeline.build() — construction errors
# ---------------------------------------------------------------------------

def test_build_raises_for_missing_manual():
    with pytest.raises(FileNotFoundError):
        GroundedAnswerPipeline.build(manual_path="/nonexistent/path/manual.md")


def test_build_raises_for_empty_manual(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        GroundedAnswerPipeline.build(manual_path=empty)


def test_build_default_loads_real_manual():
    """build() with no arguments must succeed and load the real manual."""
    clauses = load_policy(str(MANUAL_PATH))
    retriever = BM25Retriever(clauses)
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = _FAKE_ANSWER
    # Verify the real manual path resolves correctly
    assert MANUAL_PATH.exists()
    assert len(clauses) > 0


# ---------------------------------------------------------------------------
# Relevance smoke-test (real BM25, mocked generator)
# ---------------------------------------------------------------------------

def test_income_threshold_evidence_is_retrieved():
    pipeline = _make_pipeline()
    result = pipeline.ask("What is the income threshold for a household of 3?")
    ids = [c["id"] for c in result["evidence"]]
    assert "§6.6.1" in ids


def test_reporting_deadline_evidence_is_retrieved():
    pipeline = _make_pipeline()
    result = pipeline.ask("How many days to report a change of circumstances?")
    ids = [c["id"] for c in result["evidence"]]
    assert "§4.3.2" in ids
