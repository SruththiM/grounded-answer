"""
tests/test_generator.py — tests for build_prompt and AnswerGenerator.

All tests are pure — no real LLM API calls are made.
Prompt construction is tested directly via build_prompt().
AnswerGenerator.generate() is tested with a monkeypatched
OpenAI-compatible Groq client.

Run with:
    python -m pytest tests/test_generator.py -v
"""

import pytest

from src.generator import (
    build_prompt,
    AnswerGenerator,
    _NO_EVIDENCE_ANSWER,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_clauses():
    return [
        {
            "id": "§6.6.1",
            "part": 6,
            "section": "6.6",
            "heading": "Income thresholds",
            "text": (
                "**6.6.1** A household is not eligible where countable income exceeds "
                "the applicable threshold. The thresholds are —\n\n"
                "| Household size | Monthly threshold |\n"
                "|:--|:--|\n"
                "| 3 | $2,000 |"
            ),
            "score": 9.12,
        },
        {
            "id": "§7.1.1",
            "part": 7,
            "section": "7.1",
            "heading": "The award",
            "text": (
                "**7.1.1** The monthly award is the applicable needs figure "
                "for the household under §7.2, less countable income."
            ),
            "score": 4.55,
        },
    ]


@pytest.fixture
def question():
    return "What is the income threshold for a household of 3?"


# ---------------------------------------------------------------------------
# build_prompt — structure
# ---------------------------------------------------------------------------

def test_prompt_is_two_messages(question, sample_clauses):
    messages = build_prompt(question, sample_clauses)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_system_message_contains_grounding_instructions(
    question,
    sample_clauses,
):
    messages = build_prompt(question, sample_clauses)
    system = messages[0]["content"]

    assert "ONLY the policy evidence" in system
    assert "Do not use outside knowledge" in system
    assert "Do not invent" in system
    assert "insufficient" in system


def test_user_message_contains_question(question, sample_clauses):
    messages = build_prompt(question, sample_clauses)
    user = messages[1]["content"]

    assert question in user


def test_user_message_contains_clause_ids(question, sample_clauses):
    messages = build_prompt(question, sample_clauses)
    user = messages[1]["content"]

    assert "§6.6.1" in user
    assert "§7.1.1" in user


def test_user_message_contains_clause_text(question, sample_clauses):
    messages = build_prompt(question, sample_clauses)
    user = messages[1]["content"]

    assert "$2,000" in user
    assert "monthly award" in user


def test_user_message_contains_clause_headings(question, sample_clauses):
    messages = build_prompt(question, sample_clauses)
    user = messages[1]["content"]

    assert "Income thresholds" in user
    assert "The award" in user


# ---------------------------------------------------------------------------
# build_prompt — grounding
# ---------------------------------------------------------------------------

def test_full_manual_is_not_in_prompt(question, sample_clauses):
    """Only retrieved clauses should appear, not the whole manual."""

    messages = build_prompt(question, sample_clauses)

    full_text = (
        messages[0]["content"]
        + messages[1]["content"]
    )

    assert "§4.3.2" not in full_text
    assert "§9.1.4" not in full_text
    assert "§2.1.1" not in full_text


def test_only_retrieved_clause_ids_appear(question, sample_clauses):
    messages = build_prompt(question, sample_clauses)
    user = messages[1]["content"]

    assert "§6.6.1" in user
    assert "§7.1.1" in user

    assert "§4.3.2" not in user


# ---------------------------------------------------------------------------
# build_prompt — empty evidence
# ---------------------------------------------------------------------------

def test_empty_clauses_produces_no_evidence_marker(question):
    messages = build_prompt(question, [])
    user = messages[1]["content"]

    assert "No policy evidence was retrieved" in user


def test_empty_clauses_still_contains_question(question):
    messages = build_prompt(question, [])
    user = messages[1]["content"]

    assert question in user


# ---------------------------------------------------------------------------
# No hard-coded API key
# ---------------------------------------------------------------------------

def test_no_api_key_in_generator_source():
    """The generator source must not contain a hard-coded API key."""

    import inspect
    import src.generator as mod

    source = inspect.getsource(mod)

    assert "sk-" not in source


def test_generator_reads_key_from_environment(monkeypatch):
    """
    AnswerGenerator must raise EnvironmentError when GROQ_API_KEY
    is unset.
    """

    import sys
    import types

    fake_openai = types.ModuleType("openai")

    # This fake constructor accepts the same arguments as our generator.
    fake_openai.OpenAI = lambda api_key, base_url=None: None

    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai,
    )

    monkeypatch.delenv(
        "GROQ_API_KEY",
        raising=False,
    )

    gen = AnswerGenerator()

    with pytest.raises(
        EnvironmentError,
        match="GROQ_API_KEY",
    ):
        gen.generate(
            "any question",
            [
                {
                    "id": "§1.1.1",
                    "part": 1,
                    "section": "1.1",
                    "heading": "h",
                    "text": "some text",
                }
            ],
        )


# ---------------------------------------------------------------------------
# AnswerGenerator — no-evidence fast path
# ---------------------------------------------------------------------------

def test_generate_returns_insufficient_for_empty_clauses():
    gen = AnswerGenerator()

    result = gen.generate(
        "any question",
        [],
    )

    assert result == _NO_EVIDENCE_ANSWER


def test_generate_returns_insufficient_for_empty_question(sample_clauses):
    gen = AnswerGenerator()

    result = gen.generate(
        "",
        sample_clauses,
    )

    assert result == _NO_EVIDENCE_ANSWER


def test_generate_returns_insufficient_for_whitespace_question(
    sample_clauses,
):
    gen = AnswerGenerator()

    result = gen.generate(
        "   ",
        sample_clauses,
    )

    assert result == _NO_EVIDENCE_ANSWER


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------

class _FakeChoice:
    class _FakeMessage:
        content = (
            "According to §6.6.1, the threshold for a household "
            "of 3 is $2,000 per month."
        )

    message = _FakeMessage()


class _FakeCompletion:
    choices = [_FakeChoice()]


class _FakeClient:
    class chat:
        class completions:

            @staticmethod
            def create(**kwargs):
                return _FakeCompletion()


# ---------------------------------------------------------------------------
# AnswerGenerator — mocked Groq call
# ---------------------------------------------------------------------------

def test_generate_with_mocked_groq(
    monkeypatch,
    question,
    sample_clauses,
):
    """
    Full generate() path with a fake Groq client.
    No real API call is made.
    """

    monkeypatch.setenv(
        "GROQ_API_KEY",
        "test-key-not-real",
    )

    import sys
    import types

    fake_openai = types.ModuleType("openai")

    fake_openai.OpenAI = (
        lambda api_key, base_url=None: _FakeClient()
    )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai,
    )

    gen = AnswerGenerator()

    answer = gen.generate(
        question,
        sample_clauses,
    )

    assert "§6.6.1" in answer
    assert "$2,000" in answer


def test_generate_sends_only_retrieved_clauses(
    monkeypatch,
    question,
    sample_clauses,
):
    """
    Verify the prompt passed to the fake LLM contains only
    the retrieved clauses.
    """

    monkeypatch.setenv(
        "GROQ_API_KEY",
        "test-key-not-real",
    )

    captured = {}

    class _CapturingClient:
        class chat:
            class completions:

                @staticmethod
                def create(**kwargs):
                    captured["messages"] = kwargs["messages"]
                    captured["model"] = kwargs["model"]
                    captured["temperature"] = kwargs["temperature"]
                    captured["max_tokens"] = kwargs["max_tokens"]

                    return _FakeCompletion()

    import sys
    import types

    fake_openai = types.ModuleType("openai")

    fake_openai.OpenAI = (
        lambda api_key, base_url=None: _CapturingClient()
    )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai,
    )

    gen = AnswerGenerator()

    gen.generate(
        question,
        sample_clauses,
    )

    user_content = captured["messages"][1]["content"]

    assert "§6.6.1" in user_content
    assert "§7.1.1" in user_content

    assert "§4.3.2" not in user_content

    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 512