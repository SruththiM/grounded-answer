"""
generator.py — grounded answer generation over retrieved policy clauses.

Uses Groq's OpenAI-compatible API.

Environment variables:
    GROQ_API_KEY   — required
    GROQ_MODEL     — optional
                     defaults to "openai/gpt-oss-20b"
"""

import os


_SYSTEM_INSTRUCTIONS = """\
You are a policy assistant for the Calder County Household Support Program.

Rules you must follow without exception:
1. Answer using ONLY the policy evidence provided below. Do not use outside knowledge.
2. Do not invent policy rules, thresholds, dates, amounts, or exceptions.
3. If the supplied evidence does not contain enough information to answer the question,
   respond with exactly: "The available policy evidence is insufficient to answer this question."
4. Preserve the precise meaning of the policy. Do not paraphrase in a way that changes meaning.
5. Be concise. Do not repeat the question back.
6. Cite the relevant clause identifier (e.g. §6.6.1) whenever the evidence contains one.
   Format citations inline, for example: "According to §6.6.1, ..."
"""


_NO_EVIDENCE_ANSWER = (
    "The available policy evidence is insufficient to answer this question."
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