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
    """Lowercase and split on non-alphanumeric characters, dropping empty tokens."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class BM25Retriever:
    """
    Wraps a list of clause dicts (from load_policy) with a BM25Okapi index.

    The index is built once at construction time.  Repeated calls to
    retrieve() are cheap — only the query is tokenised on each call.
    """

    def __init__(self, clauses: list[dict]) -> None:
        self._clauses = clauses
        if clauses:
            corpus = [_tokenise(c["text"]) for c in clauses]
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Return the top-k clauses most relevant to *query*, ranked by BM25 score.

        Each result is a copy of the clause dict with an extra "score" key.
        Returns an empty list for an empty query or when no clauses are loaded.
        """
        if not query or not query.strip():
            return []
        if self._bm25 is None:
            return []

        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)

        k = min(top_k, len(self._clauses))
        # argsort descending — pick the top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            clause = dict(self._clauses[idx])   # shallow copy; never mutate the source
            clause["score"] = round(float(scores[idx]), 4)
            results.append(clause)

        return results


# ---------------------------------------------------------------------------
# Run:  python src/retriever.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from chunker import load_policy

    manual_path = Path(__file__).parent.parent / "data" / "policy-manual.md"
    clauses = load_policy(str(manual_path))
    retriever = BM25Retriever(clauses)

    query = "income threshold household of 3"
    print(f"Query : {query!r}\n")
    for r in retriever.retrieve(query, top_k=5):
        print(f"  {r['id']}  score={r['score']}  [{r['heading']}]")
        print(f"  {r['text'][:120].strip()}")
        print()
