from pathlib import Path

from src.chunker import load_policy
from src.retriever import BM25Retriever


MANUAL_PATH = Path(__file__).parent.parent / "data" / "policy-manual.md"


EVALUATION_CASES = [
    {
        "question": "What is the income threshold for a household of 3?",
        "expected_clause": "§6.6.1",
    },
    {
        "question": "How many days does a recipient have to report a change of circumstances?",
        "expected_clause": "§4.3.2",
    },
]


def build_retriever():
    clauses = load_policy(str(MANUAL_PATH))
    return BM25Retriever(clauses)


def test_income_threshold_retrieval():
    retriever = build_retriever()

    results = retriever.retrieve(
        EVALUATION_CASES[0]["question"],
        top_k=5,
    )

    ids = [result["id"] for result in results]

    assert EVALUATION_CASES[0]["expected_clause"] in ids


def test_reporting_deadline_retrieval():
    retriever = build_retriever()

    results = retriever.retrieve(
        EVALUATION_CASES[1]["question"],
        top_k=5,
    )

    ids = [result["id"] for result in results]

    assert EVALUATION_CASES[1]["expected_clause"] in ids