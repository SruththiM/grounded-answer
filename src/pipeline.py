"""
pipeline.py — orchestration for grounded policy answers.

Pipeline stages:
    1. BM25 retrieval
    2. Conservative temporal-context detection
    3. LLM generation using retrieved evidence only
    4. Citation extraction
    5. Status classification

Important:
- Generic policy questions do NOT require a date.
- Temporal logic activates only when the question explicitly contains
  a determination date or change-of-circumstances date.
- The pipeline never hard-codes policy answers.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.chunker import load_policy
from src.retriever import BM25Retriever
from src.generator import AnswerGenerator, _NO_EVIDENCE_ANSWER


_DEFAULT_MANUAL = Path(__file__).parent.parent / "data" / "policy-manual.md"


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

# Accept both the correctly encoded section symbol and the mojibake form
# that may appear in older test/manual text.
_SECTION_PATTERN = re.compile(
    r"(?:§|Â§)[\s\u2000-\u200b\u202f\u00a0]*"
    r"(\d+\.\d+\.\d+[A-Za-z]?)",
)

_AMENDMENT_PATTERN = re.compile(
    r"(?:Amendment\s+(?:No\.\s+)?)?2026-01"
    r"(?:\s*(?:¶|Â¶|para(?:graph)?\.?)\s*\d+(?:\.\d+)?)?",
    re.IGNORECASE,
)


def extract_citations(text: str) -> list[str]:
    """
    Extract unique clause and amendment citations while preserving order.

    Supports:
        §6.6.1
        Â§6.6.1
        §10.5.3A
        Amendment 2026-01 ¶3.1
        Amendment No. 2026-01 paragraph 3.1
    """
    if not text:
        return []

    unique: list[str] = []
    seen: set[str] = set()

    # Clause citations
    for match in _SECTION_PATTERN.finditer(text):
        clause_id = f"§{match.group(1)}"

        if clause_id not in seen:
            seen.add(clause_id)
            unique.append(clause_id)

    # Amendment citations
    for match in _AMENDMENT_PATTERN.finditer(text):
        raw = match.group(0).strip()

        # Only add amendment references that actually contain a paragraph
        # when possible; avoid treating every occurrence of "2026-01"
        # as a citation.
        if re.search(
            r"(?:¶|Â¶|para(?:graph)?\.?)\s*\d+(?:\.\d+)?",
            raw,
            re.IGNORECASE,
        ):
            paragraph_match = re.search(
                r"(?:¶|Â¶|para(?:graph)?\.?)\s*(\d+(?:\.\d+)?)",
                raw,
                re.IGNORECASE,
            )

            if paragraph_match:
                normalized = (
                    f"Amendment 2026-01 "
                    f"¶{paragraph_match.group(1)}"
                )

                if normalized not in seen:
                    seen.add(normalized)
                    unique.append(normalized)

    return unique


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

def classify_status(answer: str) -> str:
    """
    Classify the response into one of four standard states:

        contradiction
        gap
        refusal
        answered
    """
    if not answer or answer == _NO_EVIDENCE_ANSWER:
        return "refusal"

    lower = answer.lower()

    # Contradiction takes precedence over generic insufficiency/refusal.
    if (
        "conflicting provisions" in lower
        or (
            "conflict" in lower
            and (
                "provisions" in lower
                or "clause" in lower
                or "section" in lower
                or "days" in lower
            )
        )
    ):
        return "contradiction"

    if "unable to determine" in lower or "insufficient" in lower:
        if (
            "gap" in lower
            or "cross-reference" in lower
            or "does not specify" in lower
            or "not provided" in lower
            or "full-time student" in lower
            or "student" in lower
            or "no rule" in lower
        ):
            return "gap"

        return "refusal"

    return "answered"


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

_AMENDMENT_EFFECTIVE_DATE = datetime(2026, 3, 1).date()


def _parse_explicit_date(question: str):
    """
    Parse common explicit dates from the question.

    Supported examples:
        March 20, 2026
        March 20 2026
        April 15, 2026
        2026-04-15
        15 April 2026
        15th April 2026
    """
    if not question:
        return None

    # ISO format: 2026-04-15
    iso_match = re.search(
        r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b",
        question,
        re.IGNORECASE,
    )

    if iso_match:
        try:
            return datetime(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            ).date()
        except ValueError:
            pass

    months = (
        "January|February|March|April|May|June|July|August|"
        "September|October|November|December"
    )

    # Month DD, YYYY / Month DD YYYY
    month_first = re.search(
        rf"\b({months})\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?"
        r"(?:,\s*|\s+)"
        r"(20\d{2})\b",
        question,
        re.IGNORECASE,
    )

    if month_first:
        try:
            return datetime.strptime(
                f"{month_first.group(1)} "
                f"{month_first.group(2)} "
                f"{month_first.group(3)}",
                "%B %d %Y",
            ).date()
        except ValueError:
            pass

    # DD Month YYYY
    day_first = re.search(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+"
        rf"({months})\s+(20\d{{2}})\b",
        question,
        re.IGNORECASE,
    )

    if day_first:
        try:
            return datetime.strptime(
                f"{day_first.group(1)} "
                f"{day_first.group(2)} "
                f"{day_first.group(3)}",
                "%d %B %Y",
            ).date()
        except ValueError:
            pass

    return None


def _is_determination_question(question: str) -> bool:
    """
    Temporal determination logic is activated ONLY when the question
    explicitly refers to a determination/decision date.
    """
    lower = question.lower()

    determination_patterns = (
        r"\bdetermination\s+(?:made|date)\b",
        r"\bdetermination\s+on\b",
        r"\bdetermined\s+on\b",
        r"\bdecision\s+(?:made|date)\b",
        r"\bdecision\s+on\b",
        r"\bdecision\s+date\b",
        r"\baward\s+(?:made|determined)\b",
    )

    return any(
        re.search(pattern, lower)
        for pattern in determination_patterns
    )


def _is_reporting_question(question: str) -> bool:
    """
    Detect questions specifically about reporting a change of circumstances.

    This is intentionally conservative so ordinary policy questions are
    unaffected.
    """
    lower = question.lower()

    has_change = (
        "change of circumstances" in lower
        or "change in circumstances" in lower
    )

    if not has_change:
        return False

    reporting_terms = (
        "report",
        "reporting",
        "notify",
        "notification",
        "inform",
        "days",
        "deadline",
        "how long",
    )

    return any(term in lower for term in reporting_terms)


def build_temporal_context(question: str) -> str:
    """
    Build a conservative temporal directive.

    IMPORTANT:
    - Generic questions return an empty string.
    - No policy answer/value is hard-coded here.
    - Only the temporal selection rule is communicated to the generator.
    """

    if not question or not question.strip():
        return ""

    is_reporting = _is_reporting_question(question)
    is_determination = _is_determination_question(question)

    explicit_date = _parse_explicit_date(question)

    # ---------------------------------------------------------------
    # Reporting questions
    # ---------------------------------------------------------------

    if is_reporting:
        # Explicit reporting question without a date:
        # this is one of the cases where date information is necessary.
        if explicit_date is None:
            return (
                "REFUSE THIS QUESTION\n"
                "This is an explicit change-of-circumstances reporting "
                "question, but no change-of-circumstances date was supplied. "
                "The relevant date is the date the change occurred. "
                "Request that date before determining the applicable rule."
            )

        if explicit_date < _AMENDMENT_EFFECTIVE_DATE:
            return (
                "PRE-AMENDMENT RULES APPLY\n"
                "This is a change-of-circumstances reporting question. "
                "The change occurred before the amendment effective date. "
                "Use the reporting provision applicable on the date of "
                "the change. Do not apply a later reporting rule."
            )

        return (
            "AMENDMENT APPLIES\n"
            "This is a change-of-circumstances reporting question. "
            "The change occurred on or after the amendment effective date. "
            "Apply the amended reporting provision supplied in the policy "
            "evidence. The evidence is the source of truth."
        )

    # ---------------------------------------------------------------
    # Determination questions
    # ---------------------------------------------------------------

    if is_determination:
        if explicit_date is None:
            return (
                "REFUSE THIS QUESTION\n"
                "This question explicitly asks about a determination or "
                "decision date, but no determination date was supplied. "
                "Request the determination date before selecting the "
                "applicable temporal rule."
            )

        if explicit_date < _AMENDMENT_EFFECTIVE_DATE:
            return (
                "PRE-AMENDMENT RULES APPLY\n"
                "The determination date is before the amendment effective "
                "date. Use the pre-amendment provision contained in the "
                "supplied policy evidence."
            )

        return (
            "AMENDMENT APPLIES\n"
            "The determination date is on or after the amendment effective "
            "date. Apply the amended provision contained in the supplied "
            "policy evidence."
        )

    # ---------------------------------------------------------------
    # Generic questions
    # ---------------------------------------------------------------

    # CRITICAL:
    # A generic question such as:
    # "What is the income threshold for a household of 3?"
    # must proceed normally.
    return ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GroundedAnswerPipeline:
    """
    Orchestrates:

        retrieval
        temporal-context detection
        answer generation
        citation extraction
        status classification
    """

    def __init__(
        self,
        retriever: BM25Retriever,
        generator: AnswerGenerator,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    @classmethod
    def build(
        cls,
        manual_path: str | Path | None = None,
    ) -> "GroundedAnswerPipeline":
        """
        Load policy manual, build BM25 index, and construct pipeline.
        """
        path = (
            Path(manual_path)
            if manual_path
            else _DEFAULT_MANUAL
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Policy manual not found: {path}"
            )

        clauses = load_policy(str(path))

        if not clauses:
            raise ValueError(
                f"No clauses parsed from policy manual: {path}"
            )

        return cls(
            retriever=BM25Retriever(clauses),
            generator=AnswerGenerator(),
        )

    # ------------------------------------------------------------------
    # Stage 1 — Retrieval
    # ------------------------------------------------------------------

    def retrieve_evidence(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k relevant policy clauses using BM25."""
        if not question or not question.strip():
            return []

        return self._retriever.retrieve(
            question,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Stage 2 — Generation
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """
        Generate a grounded answer.

        Generic questions use the original two-argument call:

            generator.generate(question, evidence)

        Temporal questions receive an additional temporal_context keyword.
        """

        if (
            not question
            or not question.strip()
            or not evidence
        ):
            return _NO_EVIDENCE_ANSWER

        temporal_context = build_temporal_context(question)

        # Only explicit temporal questions can reach this refusal.
        # Generic questions never enter this branch.
        if "REFUSE THIS QUESTION" in temporal_context:
            return (
                "Unable to determine from the policy manual.\n\n"
                "The applicable policy depends on the relevant date, "
                "but the question does not provide enough date information "
                "to determine which provision applies.\n\n"
                "Next step: Please provide the relevant "
                "determination date or change-of-circumstances date."
            )

        # IMPORTANT:
        # Existing mocked tests expect exactly:
        #
        # generator.generate(question, evidence)
        #
        # for ordinary questions.
        if not temporal_context:
            return self._generator.generate(
                question,
                evidence,
            )

        # Temporal questions may receive the additional context.
        return self._generator.generate(
            question,
            evidence,
            temporal_context=temporal_context,
        )

    # ------------------------------------------------------------------
    # Stage 3 & 4 — Evaluation
    # ------------------------------------------------------------------

    def evaluate_decision(
        self,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> tuple[str, bool, list[str]]:
        """
        Extract citations and classify final status.
        """
        citations = extract_citations(answer)
        status = classify_status(answer)

        is_refusal = status in (
            "refusal",
            "contradiction",
            "gap",
        )

        return status, is_refusal, citations

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 5,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run the full grounded-answer pipeline.
        """

        if not question or not question.strip():
            return {
                "answer": _NO_EVIDENCE_ANSWER,
                "evidence": [],
                "citations": [],
                "status": "refusal",
                "is_refusal": True,
            }

        # Stage 1
        evidence = self.retrieve_evidence(
            question,
            top_k=top_k,
        )

        # Dry-run mode deliberately skips the generator.
        if dry_run:
            answer = "(dry-run — LLM call skipped)"

            citations = [
                c["id"]
                for c in evidence
                if "id" in c
            ]

            status = (
                "answered"
                if evidence
                else "refusal"
            )

            is_refusal = not bool(evidence)

        else:
            # Stage 2
            answer = self.generate_answer(
                question,
                evidence,
            )

            # Stage 3 & 4
            (
                status,
                is_refusal,
                citations,
            ) = self.evaluate_decision(
                answer,
                evidence,
            )

        return {
            "answer": answer,
            "evidence": evidence,
            "citations": citations,
            "status": status,
            "is_refusal": is_refusal,
        }