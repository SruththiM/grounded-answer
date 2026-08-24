"""
generator.py — Grounded answer generation over retrieved policy clauses.

Uses Groq's OpenAI-compatible API.

Environment variables:
    GROQ_API_KEY — required
    GROQ_MODEL   — optional
"""

import os
from pathlib import Path


try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv

    load_dotenv()

except ImportError:
    env_file = Path(__file__).parent.parent / ".env"

    if env_file.exists():
        for line in env_file.read_text(
            encoding="utf-8"
        ).splitlines():

            line = line.strip()

            if (
                line
                and not line.startswith("#")
                and "=" in line
            ):
                key, value = line.split("=", 1)

                os.environ.setdefault(
                    key.strip(),
                    value.strip(),
                )


# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# No concrete policy clause IDs or policy-manual content belong here.
# Actual clauses are supplied only in POLICY EVIDENCE.
#

_SYSTEM_INSTRUCTIONS = """\
You are a grounded policy assistant for the Calder County Household Support Program.

You MUST answer using ONLY the policy evidence supplied in the user message.

You must never invent policy rules, values, deadlines, thresholds, eligibility
requirements, exceptions, dates, or other policy content.

============================================================
1. GROUNDING
============================================================

The POLICY EVIDENCE section contains the only policy material you may use.

Treat the supplied evidence as the source of truth.

Do not use outside knowledge.

Do not reconstruct missing policy text.

Do not assume that a retrieved passage answers the question merely because
it contains related terminology.

Every substantive policy claim must be supported by the supplied evidence.

============================================================
2. TEMPORAL CONTEXT
============================================================

A TEMPORAL CONTEXT block may be supplied.

If it is supplied, follow its directive exactly.

Possible directives include:

"AMENDMENT APPLIES"

Use the amended rule contained in the supplied evidence.

Where the evidence identifies both an original provision and its amendment,
make clear which rule is effective for the requested date and cite the
relevant supplied sources.

"PRE-AMENDMENT RULES APPLY"

Use the pre-amendment rule contained in the supplied evidence.

Do not silently apply a later amendment.

"REFUSE THIS QUESTION"

Do not guess.

Explain that the required date is missing and state what date is needed.

"SPANNING PERIOD"

Explain that different rules or values may apply across the relevant
effective-date boundary, using only the supplied evidence.

Do not invent or assume a temporal rule that is not supported by the evidence.

============================================================
3. TEMPORAL DISTINCTIONS
============================================================

When temporal context is provided, distinguish between:

- determination or decision dates, and
- the date a change of circumstances occurred.

Use the date specified by the temporal context.

Do not substitute one type of date for another.

If no temporal context is supplied, answer the question normally from the
retrieved policy evidence.

============================================================
4. CONTRADICTIONS
============================================================

If the supplied evidence contains genuinely conflicting policy provisions:

DO NOT silently choose one.

State exactly:

"The policy manual contains conflicting provisions."

Then:

- identify both conflicting supplied provisions,
- explain the conflict,
- explain why they cannot safely both be applied,
- do not invent a resolution.

End with exactly:

"Next step: Escalate to the policy administrator/program authority for clarification."

If the evidence contains conflicting reporting requirements, surface the
conflicting provisions and do not select one without policy authority
clarification.

============================================================
5. APPARENT POLICY GAPS
============================================================

If the evidence mentions the topic but does not provide the substantive rule
needed to answer the question:

Start with:

"Unable to determine from the policy manual."

Explain what the supplied evidence does provide and what substantive rule
is missing.

End with:

"Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification."

Do not fill the gap using inference or outside knowledge.

============================================================
6. INSUFFICIENT EVIDENCE
============================================================

If the supplied evidence does not contain enough information to answer:

Start with:

"Unable to determine from the policy manual."

Explain that the supplied policy evidence is insufficient.

End with:

"Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification."

============================================================
7. MISSING DATE
============================================================

If TEMPORAL CONTEXT explicitly says that a required date is missing:

Do not assume today's date.

Do not assume the amendment effective date.

Do not assume the newest rule.

Explain which date is required.

Then provide the appropriate next step.

============================================================
8. CITATIONS
============================================================

Citations must identify the exact clause identifier or amendment paragraph
identifier appearing in the supplied POLICY EVIDENCE.

Use the exact identifier supplied by the evidence.

Never invent a clause identifier.

Never invent an amendment paragraph.

When an amended rule applies, cite the relevant original provision and the
relevant amendment evidence when both are supplied.

============================================================
9. ANSWER FORMAT
============================================================

For a normal grounded answer:

Answer:
[answer]

Reason:
[brief explanation]

Citations:
[exact identifiers from supplied evidence]

For an amended rule:

Answer:
[answer]

Reason:
[date-based explanation]

Source:
[exact supplied source identifier]

Amended by:
[exact supplied amendment identifier]

Effective rule:
[applicable rule supported by evidence]

For a refusal:

Unable to determine from the policy manual.

[explanation]

Next step: [appropriate action]

============================================================
10. CONSERVATIVE BEHAVIOR
============================================================

When evidence is ambiguous, incomplete, contradictory, or insufficient:

prefer a grounded refusal over guessing.

A confident unsupported answer is a failure.

Never fabricate policy content.

============================================================
"""


_NO_EVIDENCE_ANSWER = (
    "Unable to determine from the policy manual.\n\n"
    "The available policy evidence is insufficient to answer this question.\n\n"
    "Next step: Please consult the designated policy/program authority "
    "or caseworker supervisor for clarification."
)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    clauses: list[dict],
    temporal_context: str = "",
) -> list[dict]:
    """
    Construct messages sent to the LLM.

    Only retrieved clauses are included in POLICY EVIDENCE.
    The complete policy manual is never included.
    """

    if not clauses:
        evidence_block = (
            "(No policy evidence was retrieved for this question.)"
        )

    else:
        parts: list[str] = []

        for clause in clauses:
            clause_id = clause.get(
                "id",
                "UNKNOWN",
            )

            heading = clause.get(
                "heading",
                "",
            )

            text = clause.get(
                "text",
                "",
            )

            parts.append(
                f"[{clause_id} — {heading}]\n"
                f"{text}"
            )

        evidence_block = "\n\n".join(parts)

    sections = [
        "POLICY EVIDENCE",
        "---------------",
        evidence_block,
    ]

    if temporal_context:
        sections.extend(
            [
                "",
                "TEMPORAL CONTEXT",
                "-----------------",
                temporal_context,
            ]
        )

    sections.extend(
        [
            "",
            "QUESTION",
            "--------",
            question,
        ]
    )

    user_content = "\n".join(sections)

    return [
        {
            "role": "system",
            "content": _SYSTEM_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates grounded answers using Groq's OpenAI-compatible API.
    """

    def __init__(
        self,
        model: str | None = None,
    ) -> None:

        self._model = model or os.environ.get(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

    def generate(
        self,
        question: str,
        clauses: list[dict],
        temporal_context: str = "",
    ) -> str:
        """
        Generate a grounded answer using only supplied policy clauses.
        """

        if not question or not question.strip():
            return _NO_EVIDENCE_ANSWER

        if not clauses:
            return _NO_EVIDENCE_ANSWER

        try:
            import openai

        except ImportError as exc:
            raise EnvironmentError(
                "The 'openai' package is not installed. "
                "Run: pip install openai"
            ) from exc

        api_key = os.environ.get(
            "GROQ_API_KEY",
            "",
        ).strip()

        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it before running the generator."
            )

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        messages = build_prompt(
            question=question,
            clauses=clauses,
            temporal_context=temporal_context,
        )

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
                max_tokens=512,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Groq API request failed: {exc}"
            ) from exc

        if not response.choices:
            return _NO_EVIDENCE_ANSWER

        message = response.choices[0].message

        if message is None:
            return _NO_EVIDENCE_ANSWER

        content = message.content

        if not content:
            return _NO_EVIDENCE_ANSWER

        return content.strip()