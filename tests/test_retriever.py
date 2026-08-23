"""
tests/test_retriever.py — tests for BM25Retriever.

Run with:  pytest tests/test_retriever.py -v
"""

from pathlib import Path

import pytest

from src.chunker import load_policy
from src.retriever import BM25Retriever, _tokenise

MANUAL_PATH = Path(__file__).parent.parent / "data" / "policy-manual.md"


@pytest.fixture(scope="module")
def retriever():
    clauses = load_policy(str(MANUAL_PATH))
    return BM25Retriever(clauses)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def test_tokenise_lowercases():
    assert _tokenise("Hello World") == ["hello", "world"]


def test_tokenise_strips_punctuation():
    assert _tokenise("§4.3.2, income!") == ["4", "3", "2", "income"]


def test_tokenise_empty():
    assert _tokenise("") == []


# ---------------------------------------------------------------------------
# Construction edge cases
# ---------------------------------------------------------------------------

def test_empty_clause_list_returns_empty():
    r = BM25Retriever([])
    assert r.retrieve("anything") == []


# ---------------------------------------------------------------------------
# Basic retrieval
# ---------------------------------------------------------------------------

def test_retrieve_returns_list(retriever):
    results = retriever.retrieve("income threshold household of 3")
    assert isinstance(results, list)


def test_retrieve_default_top_k(retriever):
    results = retriever.retrieve("income threshold household of 3")
    assert len(results) == 5


def test_retrieve_custom_top_k(retriever):
    results = retriever.retrieve("income threshold", top_k=3)
    assert len(results) == 3


def test_retrieve_top_k_larger_than_corpus():
    """top_k > number of clauses should return all clauses, not raise."""
    tiny = [
        {"id": "§1.1.1", "part": 1, "section": "1.1", "heading": "h", "text": "foo bar"},
        {"id": "§1.1.2", "part": 1, "section": "1.1", "heading": "h", "text": "baz qux"},
    ]
    r = BM25Retriever(tiny)
    results = r.retrieve("foo", top_k=100)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

def test_income_threshold_clause_is_retrieved(retriever):
    """§6.6.1 holds the income threshold table and must surface for this query."""
    results = retriever.retrieve("income threshold household of 3", top_k=5)
    ids = [r["id"] for r in results]
    assert "§6.6.1" in ids


def test_reporting_deadline_clause_is_retrieved(retriever):
    """§4.3.2 states the change-of-circumstances reporting deadline."""
    results = retriever.retrieve("how many days to report change of circumstances", top_k=5)
    ids = [r["id"] for r in results]
    assert "§4.3.2" in ids


def test_results_are_sorted_by_score_descending(retriever):
    results = retriever.retrieve("overpayment recovery deduction", top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

def test_result_contains_required_keys(retriever):
    results = retriever.retrieve("eligibility conditions", top_k=1)
    assert len(results) == 1
    r = results[0]
    for key in ("id", "part", "section", "heading", "text", "score"):
        assert key in r, f"missing key: {key}"


def test_retrieve_does_not_mutate_source_clauses(retriever):
    """Adding 'score' to results must not bleed back into the retriever's internal list."""
    results = retriever.retrieve("sanction", top_k=1)
    assert "score" in results[0]
    # The internal clause list must not have a 'score' key
    for clause in retriever._clauses:
        assert "score" not in clause


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty(retriever):
    assert retriever.retrieve("") == []


def test_whitespace_only_query_returns_empty(retriever):
    assert retriever.retrieve("   ") == []


def test_repeated_calls_are_consistent(retriever):
    r1 = retriever.retrieve("suspension termination", top_k=5)
    r2 = retriever.retrieve("suspension termination", top_k=5)
    assert [r["id"] for r in r1] == [r["id"] for r in r2]


def test_punctuation_and_case_do_not_break_retrieval(retriever):
    r1 = retriever.retrieve("INCOME THRESHOLD", top_k=3)
    r2 = retriever.retrieve("income threshold", top_k=3)
    assert [r["id"] for r in r1] == [r["id"] for r in r2]
