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

    BM25 matches words literally. Natural language user queries may use terms
    like "away from county", "car", "salary", or "report deadline", while the
    manual uses specific terminology like "absence", "motor vehicle",
    "earnings", or "change of circumstances".

    Query expansion bridges this gap without modifying the original question.
    """
    expanded = query

    # Eligibility temporarily stopping / suspension
    if re.search(r"\btemporarily\s+ineligible\b", query, flags=re.IGNORECASE):
        expanded += " suspension suspended award"

    # Temporary absence / medical absence
    if re.search(r"\b(away|left|out\s+of|leave)\s+(the\s+)?county\b", query, flags=re.IGNORECASE):
        expanded += " absence absent temporary residence"

    # Vehicle / car
    if re.search(r"\b(car|automobile|truck|vehicle)\b", query, flags=re.IGNORECASE):
        expanded += " motor vehicle countable resources"

    # Earnings / salary / employment
    if re.search(r"\b(salary|wages|job|work\s+income|earnings|disregard)\b", query, flags=re.IGNORECASE):
        expanded += " earnings employment disregard countable income 120 175 amendment"

    # Change of circumstances reporting / overpayments
    if re.search(r"\b(report|reporting|notify|deadline|how\s+many\s+days)\b", query, flags=re.IGNORECASE) and re.search(r"\b(change|circumstances)\b", query, flags=re.IGNORECASE):
        expanded += " calendar days overpayment obligations 10 14 30 amendment"

    # Income thresholds
    if re.search(r"\b(income\s+threshold|threshold|monthly\s+threshold)\b", query, flags=re.IGNORECASE):
        expanded += " income thresholds monthly threshold 2410 2500 amendment"

    # Full-time students / higher education
    if re.search(r"\b(student|students|college|university|higher\s+education)\b", query, flags=re.IGNORECASE):
        expanded += " full time student education enrolment needs figure"

    # Minimum award threshold / small entitlement ($25 floor)
    if re.search(r"\b(award|entitlement|paid)\b", query, flags=re.IGNORECASE) and re.search(r"\b(calculated|deducting|\$\d+|less|minimum|resulting)\b", query, flags=re.IGNORECASE):
        expanded += " resulting figure less than 25 no award is made"

    # Sanctions / exceptions
    if re.search(r"\b(sanction|sanctions|penalty|penalties)\b", query, flags=re.IGNORECASE):
        expanded += " reduction monthly award 20 15 per cent dependent child activities daily living increased the award 10 5 3a amendment"

    # Activities of daily living / disability / adjustments
    if re.search(r"\b(activities\s+of\s+daily\s+living|adl|assistance)\b", query, flags=re.IGNORECASE):
        expanded += " needs figure increased assessed requiring assistance"

    # Temporal references (dates / amendment)
    if re.search(r"\b(february|march|april|2026|amendment|effective|determination|transitional)\b", query, flags=re.IGNORECASE):
        expanded += " effective 1 march 2026 amendment transitional provision"

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