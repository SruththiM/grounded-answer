import re
from pathlib import Path
from typing import Any

from src.chunker import load_policy
from src.retriever import BM25Retriever
from src.generator import AnswerGenerator, _NO_EVIDENCE_ANSWER

_DEFAULT_MANUAL = Path(__file__).parent.parent / "data" / "policy-manual.md"
_CITATION_PATTERN = re.compile(r"§[\s\u2000-\u200b\u202f\u00a0]*(\d+\.\d+\.\d+)")


def extract_citations(text: str) -> list[str]:
    """Extract unique clause citation strings (e.g. ['§6.6.1', '§4.3.2']) preserving order."""
    if not text:
        return []
    matches = _CITATION_PATTERN.findall(text)
    seen = set()
    unique = []
    for m in matches:
        norm = f"§{m}"
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def classify_status(answer: str) -> str:
    """
    Classify the response into one of four standard states:
    - 'contradiction' : Policy contains internal conflicting provisions
    - 'gap'           : Topic referenced/mentioned but operative rule missing
    - 'refusal'       : Out of scope or insufficient evidence
    - 'answered'      : Grounded policy answer provided
    """
    if not answer or answer == _NO_EVIDENCE_ANSWER:
        return "refusal"

    lower = answer.lower()
    if "conflicting provisions" in lower or ("conflict" in lower and ("provisions" in lower or "clause" in lower or "section" in lower or "days" in lower)):
        return "contradiction"

    if "unable to determine" in lower or "insufficient" in lower:
        if "gap" in lower or "cross-reference" in lower or "does not specify" in lower or "not provided" in lower or "full-time student" in lower or "student" in lower or "no rule" in lower:
            return "gap"
        return "refusal"

    return "answered"


class GroundedAnswerPipeline:
    """
    Orchestrates retrieval, grounded answer generation, citation extraction,
    and refusal/contradiction classification.

    Separates stages cleanly for Day-Two agility:
        1. Retrieval (retrieve_evidence)
        2. Answer Construction (generate_answer)
        3. Citation Extraction (extract_citations)
        4. Decision / Status Evaluation (evaluate_decision)
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

    def retrieve_evidence(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Stage 1: Retrieve top-k relevant policy clauses."""
        if not question or not question.strip():
            return []
        return self._retriever.retrieve(question, top_k=top_k)

    def generate_answer(self, question: str, evidence: list[dict[str, Any]]) -> str:
        """Stage 2: Construct grounded answer using only retrieved evidence."""
        if not question or not question.strip() or not evidence:
            return _NO_EVIDENCE_ANSWER
        return self._generator.generate(question, evidence)

    def evaluate_decision(self, answer: str, evidence: list[dict[str, Any]]) -> tuple[str, bool, list[str]]:
        """Stage 3 & 4: Extract citations and classify outcome status."""
        citations = extract_citations(answer)
        status = classify_status(answer)
        is_refusal = status in ("refusal", "contradiction", "gap")
        return status, is_refusal, citations

    def ask(self, question: str, top_k: int = 5, dry_run: bool = False) -> dict[str, Any]:
        """
        Run the full pipeline for *question*.

        Returns a dict with:
            answer          : str        — grounded answer (or structured refusal)
            evidence        : list[dict] — clause dicts retrieved by BM25
            citations       : list[str]  — clause IDs cited in the answer
            status          : str        — 'answered' | 'refusal' | 'contradiction' | 'gap'
            is_refusal      : bool       — True if the response is any type of refusal

        When dry_run=True the retrieval step runs normally but the LLM call
        is skipped — useful for inspecting retrieved evidence without an API key.
        """
        if not question or not question.strip():
            return {
                "answer": _NO_EVIDENCE_ANSWER,
                "evidence": [],
                "citations": [],
                "status": "refusal",
                "is_refusal": True,
            }

        evidence = self.retrieve_evidence(question, top_k=top_k)

        if dry_run:
            answer = "(dry-run — LLM call skipped)"
            citations = [c["id"] for c in evidence]
            status = "answered" if evidence else "refusal"
            is_refusal = not bool(evidence)
        else:
            answer = self.generate_answer(question, evidence)
            status, is_refusal, citations = self.evaluate_decision(answer, evidence)

        return {
            "answer": answer,
            "evidence": evidence,
            "citations": citations,
            "status": status,
            "is_refusal": is_refusal,
        }


