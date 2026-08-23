"""
pipeline.py — end-to-end orchestration for the grounded-answer system.

Wires together the three existing components:

    load_policy  (chunker.py)
         ↓
    BM25Retriever  (retriever.py)
         ↓
    AnswerGenerator  (generator.py)

Public interface:

    pipeline = GroundedAnswerPipeline.build()
    result   = pipeline.ask("What is the income threshold for a household of 3?")

    result["answer"]   — the grounded answer string
    result["evidence"] — list of clause dicts used to produce the answer
"""

from pathlib import Path

from src.chunker import load_policy
from src.retriever import BM25Retriever
from src.generator import AnswerGenerator, _NO_EVIDENCE_ANSWER

_DEFAULT_MANUAL = Path(__file__).parent.parent / "data" / "policy-manual.md"


class GroundedAnswerPipeline:
    """
    Orchestrates retrieval and generation.

    Construction is separated from asking so the BM25 index is built once
    and reused across multiple questions.
    """

    def __init__(self, retriever: BM25Retriever, generator: AnswerGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    @classmethod
    def build(cls, manual_path: str | Path | None = None) -> "GroundedAnswerPipeline":
        """
        Load the policy manual, build the BM25 index, and return a ready pipeline.

        Raises FileNotFoundError if the manual cannot be found.
        Raises ValueError if the manual produces no clauses.
        """
        path = Path(manual_path) if manual_path else _DEFAULT_MANUAL
        if not path.exists():
            raise FileNotFoundError(f"Policy manual not found: {path}")

        clauses = load_policy(str(path))
        if not clauses:
            raise ValueError(f"No clauses parsed from policy manual: {path}")

        return cls(
            retriever=BM25Retriever(clauses),
            generator=AnswerGenerator(),
        )

    def ask(self, question: str, top_k: int = 5, dry_run: bool = False) -> dict:
        """
        Run the full pipeline for *question*.

        Returns a dict with:
            answer   : str   — grounded answer (or insufficiency message)
            evidence : list  — clause dicts retrieved by BM25 (may be empty)

        When dry_run=True the retrieval step runs normally but the LLM call
        is skipped — useful for inspecting retrieved evidence without an API key.

        Never raises on empty question or empty retrieval — returns the
        insufficiency message instead.
        """
        if not question or not question.strip():
            return {"answer": _NO_EVIDENCE_ANSWER, "evidence": []}

        evidence = self._retriever.retrieve(question, top_k=top_k)

        if dry_run:
            answer = "(dry-run — LLM call skipped)"
        else:
            answer = self._generator.generate(question, evidence)

        return {"answer": answer, "evidence": evidence}
