"""
generator.py — grounded answer generation over retrieved policy clauses.

Architecture:

    User Question
          ↓
    BM25Retriever  (src/retriever.py)
          ↓
    top-k clauses
          ↓
    build_prompt()          ← pure function, fully unit-testable
          ↓
    AnswerGenerator.generate()  ← calls the LLM
          ↓
    Grounded answer string

Configuration (environment variables):
    OPENAI_API_KEY   — required for live calls
    OPENAI_MODEL     — optional, defaults to "gpt-4o-mini"

Typical usage:

    from src.chunker import load_policy
    from src.retriever import BM25Retriever
    from src.generator import AnswerGenerator

    clauses  = load_policy("data/policy-manual.md")
    retriever = BM25Retriever(clauses)
    generator = AnswerGenerator()

    question = "What is the income threshold for a household of 3?"
    evidence = retriever.retrieve(question, top_k=5)
    answer   = generator.generate(question, evidence)
    print(answer)
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
    Construct the messages list to send to the LLM.

    Returns a list of dicts in OpenAI chat format:
        [{"role": "system", "content": ...},
         {"role": "user",   "content": ...}]

    This function is pure — it makes no network calls and can be tested freely.
    The entire policy manual is never included; only the supplied clauses are used.
    """
    if not clauses:
        evidence_block = "(No policy evidence was retrieved for this question.)"
    else:
        parts = []
        for c in clauses:
            parts.append(f"[{c['id']} — {c['heading']}]\n{c['text']}")
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
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user",   "content": user_content},
    ]


class AnswerGenerator:
    """
    Sends a grounded prompt to the LLM and returns the answer string.

    Prompt construction is delegated to build_prompt() so it can be
    tested independently without any API calls.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, question: str, clauses: list[dict]) -> str:
        """
        Generate a grounded answer for *question* using only *clauses* as evidence.

        Returns _NO_EVIDENCE_ANSWER immediately if clauses is empty, avoiding
        an unnecessary API call.
        """
        if not question or not question.strip():
            return _NO_EVIDENCE_ANSWER

        if not clauses:
            return _NO_EVIDENCE_ANSWER

        import openai  # imported here so the module loads without openai installed

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before running the generator."
            )

        client = openai.OpenAI(api_key=api_key)
        messages = build_prompt(question, clauses)

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,        # deterministic — policy Q&A, not creative writing
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
