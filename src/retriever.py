"""
retriever.py — BM25 retrieval over policy manual clauses.

Typical usage:

    from src.chunker import load_policy
    from src.retriever import BM25Retriever

    clauses = load_policy("data/policy-manual.md")
    retriever = BM25Retriever(clauses)
    results = retriever.retrieve("income threshold household of 3", top_k=5)

    for r in results:
        print(r["id"], r["score"])
        print(r["text"])
"""

import re
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenise(text: str) -> list[str]:
    """
    Lowercase and split on non-alphanumeric characters,
    dropping empty tokens.
    """
    return [
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if token
    ]


def _expand_query(query: str) -> str:
    """
    Add a small number of policy-domain synonyms to the query.

    BM25 matches words literally. For example, the user may ask:

        "What happens if a household becomes temporarily ineligible?"

    while the policy may describe the same concept using:

        "suspension" / "suspended"

    Query expansion lets BM25 retrieve the relevant suspension
    clauses without changing the original user question.

    Only closely related policy terminology is expanded here.
    """

    expanded = query

    # Eligibility temporarily stopping is expressed as "suspension"
    # in the policy manual.
    if re.search(
        r"\btemporarily\s+ineligible\b",
        query,
        flags=re.IGNORECASE,
    ):
        expanded += " suspension suspended"

    # Similar wording users may use.
    if re.search(
        r"\btemporarily\s+ineligible\b",
        query,
        flags=re.IGNORECASE,
    ):
        expanded += " award"

    return expanded


class BM25Retriever:
    """
    Wraps a list of clause dicts from load_policy with a BM25Okapi index.

    The index is built once at construction time.

    Repeated calls to retrieve() only need to:
        1. expand the query
        2. tokenise it
        3. calculate BM25 scores
    """

    def __init__(self, clauses: list[dict]) -> None:
        self._clauses = clauses

        if clauses:
            corpus = [
                _tokenise(c["text"])
                for c in clauses
            ]

            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k clauses most relevant to *query*.

        Each result is a copy of the original clause dictionary
        with an additional "score" field.

        Returns an empty list when:
            - query is empty
            - no clauses are loaded
        """

        if not query or not query.strip():
            return []

        if self._bm25 is None:
            return []

        # Expand policy terminology before BM25 tokenisation.
        expanded_query = _expand_query(query)

        tokens = _tokenise(expanded_query)

        scores = self._bm25.get_scores(tokens)

        k = min(top_k, len(self._clauses))

        # Sort indices by score, highest first.
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results: list[dict[str, Any]] = []

        for idx in top_indices:
            # Shallow copy so the original clause is never modified.
            clause = dict(self._clauses[idx])

            clause["score"] = round(
                float(scores[idx]),
                4,
            )

            results.append(clause)

        return results


# ---------------------------------------------------------------------------
# Run:
#
#     python src/retriever.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    from chunker import load_policy

    manual_path = (
        Path(__file__).parent.parent
        / "data"
        / "policy-manual.md"
    )

    clauses = load_policy(str(manual_path))
    retriever = BM25Retriever(clauses)

    query = "income threshold household of 3"

    print(f"Query : {query!r}\n")

    for result in retriever.retrieve(
        query,
        top_k=5,
    ):
        print(
            f"  {result['id']} "
            f"score={result['score']} "
            f"[{result['heading']}]"
        )

        print(
            f"  {result['text'][:120].strip()}"
        )

        print()