"""
generator.py — grounded answer generation over retrieved policy clauses.

Uses Groq's OpenAI-compatible API.

Environment variables:
    GROQ_API_KEY   — required
    GROQ_MODEL     — optional
                     defaults to "openai/gpt-oss-20b"
"""

import os
from dotenv import load_dotenv

load_dotenv()


_SYSTEM_INSTRUCTIONS = """\
You are a grounded policy assistant for the Calder County Household Support Program.

You must strictly obey these operational rules:

1. MANDATORY CONTRADICTION CHECK:
   Before providing a policy rule or deadline, examine all retrieved clauses for conflicting, contradictory, or inconsistent provisions (for example, if one clause states a requirement of 10 calendar days while another clause references or states 30 calendar days for reporting changes).
   If conflicting provisions exist in the evidence on the matter asked:
   - You MUST NOT silently pick one clause over the other.
   - Start immediately with: "The policy manual contains conflicting provisions."
   - State and cite both conflicting clauses (e.g. "Clause §A.B.C states ... whereas Clause §D.E.F states ...").
   - Explain that these provisions cannot both be applied and the manual does not provide a consistent basis to determine the outcome.
   - End with: "Next step: Escalate to the policy administrator/program authority for clarification."

2. APPARENT POLICY GAPS / BROKEN REFERENCES:
   If a topic is mentioned or cross-referenced in a clause (e.g. a clause refers to another section for full-time students) but the substantive operative rule or formula is not provided in the manual:
   - Refuse to guess or fill the gap from outside knowledge.
   - Start with: "Unable to determine from the policy manual."
   - Cite the clause mentioning the reference and explain that the manual lacks the substantive rule.
   - End with: "Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification."

3. OUT OF SCOPE / INSUFFICIENT EVIDENCE:
   If the evidence does not contain information to answer the question:
   - Start with: "Unable to determine from the policy manual."
   - State that the supplied evidence does not contain information to answer this question.
   - End with: "Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification."

4. GROUNDED ANSWERS:
   Answer using ONLY the policy evidence provided. Every substantive claim must cite its specific clause (e.g. §X.Y.Z). Do not invent thresholds, amounts, deadlines, or exceptions.
"""


_NO_EVIDENCE_ANSWER = (
    "Unable to determine from the policy manual.\n\n"
    "The available policy evidence is insufficient to answer this question.\n\n"
    "Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification."
)


def build_prompt(question: str, clauses: list[dict]) -> list[dict]:
    """
    Construct the messages sent to the LLM.

    Only the retrieved policy clauses are included.
    The entire policy manual is never sent to the LLM.
    """

    if not clauses:
        evidence_block = (
            "(No policy evidence was retrieved for this question.)"
        )
    else:
        parts = []

        for clause in clauses:
            parts.append(
                f"[{clause['id']} — {clause['heading']}]\n"
                f"{clause['text']}"
            )

        evidence_block = "\n\n".join(parts)

    user_content = (
        "POLICY EVIDENCE\n"
        "---------------\n"
        f"{evidence_block}\n\n"
        "QUESTION\n"
        "--------\n"
        f"{question}"
    )

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


class AnswerGenerator:
    """
    Generates grounded answers using Groq's
    OpenAI-compatible API.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

    def generate(
        self,
        question: str,
        clauses: list[dict],
    ) -> str:
        """
        Generate a grounded answer using only the supplied
        policy clauses.

        No API call is made when the question or evidence is empty.
        """

        # ---------------------------------------------------------
        # Fast path: empty question
        # ---------------------------------------------------------

        if not question or not question.strip():
            return _NO_EVIDENCE_ANSWER

        # ---------------------------------------------------------
        # Fast path: no retrieved evidence
        # ---------------------------------------------------------

        if not clauses:
            return _NO_EVIDENCE_ANSWER

        # ---------------------------------------------------------
        # Import OpenAI-compatible client
        # ---------------------------------------------------------

        import openai

        # ---------------------------------------------------------
        # Read Groq API key from environment
        # ---------------------------------------------------------

        api_key = os.environ.get(
            "GROQ_API_KEY",
            "",
        ).strip()

        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it before running the generator."
            )

        # ---------------------------------------------------------
        # Create Groq client
        # ---------------------------------------------------------

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        # ---------------------------------------------------------
        # Build grounded prompt
        # ---------------------------------------------------------

        messages = build_prompt(
            question,
            clauses,
        )

        # ---------------------------------------------------------
        # Call Groq
        # ---------------------------------------------------------

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
            max_tokens=512,
        )

        # ---------------------------------------------------------
        # Extract answer
        # ---------------------------------------------------------

        content = response.choices[0].message.content

        if not content:
            return _NO_EVIDENCE_ANSWER

        return content.strip()