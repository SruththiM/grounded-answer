# AI Usage Disclosure — Grounded Answer (Calder County HSP)

This document discloses the use of Artificial Intelligence (AI) tools and assistants during the design, development, debugging, testing, and evaluation of the Grounded Answer project for the Brite Spark 2026 challenge.

---

## 1. Tools and Models Used

| Tool / Assistant | Platform | Primary Purpose |
|:---|:---|:---|
| **Antigravity IDE** | Google DeepMind (Gemini 3.7 Flash) | Repository inspection, architecture refinement, test suite completion, evaluation harness implementation, contradiction & gap analysis, documentation authoring. |
| **ChatGPT** | OpenAI / VS Code Extension | Initial milestone prototyping (chunker structure, initial BM25 retrieval, pipeline skeleton). |
| **Groq API** (`openai/gpt-oss-20b`) | Groq Cloud | Runtime inference engine for grounded answer generation. |

---

## 2. Categories of AI Assistance

### A. Repository & Corpus Analysis
- Assisted in systematic analysis of `data/policy-manual.md` to discover:
  - **The Genuine Contradiction**: Conflict between **§4.3.2** (10 calendar days reporting deadline) and **§9.1.4** (30 calendar days reporting deadline for overpayments).
  - **The Apparent Policy Gap**: Broken reference in **§7.1.3** pointing to **§5.4** for full-time students, where §5.4 only covers care allowances.

### B. Prompt Engineering & Grounding Constraints
- Formulated the zero-shot system instructions in `src/generator.py` enforcing:
  - Strict grounding only in supplied policy evidence.
  - Mandatory clause-level citations (`§X.Y.Z`).
  - Actionable refusal responses including a `"Next step:"` directive.
  - Explicit contradiction formatting (declaring internal conflict, citing both clauses without choosing, and escalating).
  - Explicit gap refusal when operative policy rules are missing.

### C. Retrieval & Query Expansion
- Designed regex-based query expansion patterns in `src/retriever.py` to map colloquial language (e.g., "away from county", "car", "salary", "minimum award", "activities of daily living") to statutory manual terms ("absence", "motor vehicle", "earnings", "resulting figure less than $25", "requiring assistance").

### D. Evaluation Suite & Test Harness
- Authored the 12-question evaluation suite in `src/evaluator.py` and `evaluate.py` spanning 7 distinct categories:
  - Answerable table lookups
  - Earnings disregards
  - Resource limits and exclusions
  - Needs figures and schedules
  - Specific adjustments
  - Paraphrased queries
  - Edge cases and sanction exemptions
  - Minimum award thresholds
  - Genuine internal contradictions
  - Apparent policy gaps
  - Out-of-scope refusals
  - Unsupported inference refusals
- Implemented automated pytest suites in `tests/test_retriever.py`, `tests/test_generator.py`, `tests/test_pipeline.py`, and `tests/test_evaluation.py`.

### E. Debugging & Compatibility
- Identified and resolved a Windows console Unicode encoding issue (`UnicodeEncodeError` with checkmark characters on CP1252) by configuring stdout to UTF-8 with safe ASCII fallbacks.
- Corrected sample clause identifier leakage in generator tests to ensure prompt isolation.
- Tuned BM25 query expansion for $25 minimum award threshold retrieval.

---

## 3. Human Review and Verification

Every piece of code, test case, evaluation metric, and documentation was subject to human verification:
1. **Manual Inspection**: The full consolidated text of `data/policy-manual.md` was inspected line-by-line to confirm the factual existence of the contradiction and gap.
2. **Deterministic Code Review**: All regex patterns, chunking boundaries, and classification logic were reviewed for edge cases and correctness.
3. **Execution Verification**: Automated tests (`pytest`) and evaluation runs (`evaluate.py --dry-run` and live `evaluate.py`) were executed and verified against real model outputs.
4. **Security Audit**: Verified that `.env` is ignored by Git, no API keys or secrets are committed, and `.env.example` provides clean template variables.
