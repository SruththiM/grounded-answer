# Grounded Answer — Calder County Household Support Program (HSP)

A fully grounded, verifiable AI Question-Answering and Retrieval-Augmented Generation (RAG) assistant for the Calder County Household Support Program policy manual.

Every substantive claim is strictly traceable to specific policy clauses (e.g., `§6.6.1`), and the system explicitly refuses or escalates when faced with out-of-scope questions, apparent policy gaps, or genuine internal contradictions.

---

## Features

- **Clause-Level Grounded Answers**: Answers strictly derived from retrieved policy manual text with explicit clause citations (e.g., `§6.6.1`, `§2.4.2`).
- **Defensible Refusal Path**: Explicitly refuses to answer when information is absent, insufficient, or ambiguous, always providing an actionable `"Next step:"` directive for caseworkers.
- **Genuine Contradiction Handling**: Detects internal conflicts in the corpus (specifically the reporting deadline discrepancy between **§4.3.2** and **§9.1.4**), presents both provisions neutrally without silently picking a side, and routes to policy administration.
- **Apparent Policy Gap Handling**: Identifies broken policy cross-references (such as **§7.1.3** pointing to **§5.4** for full-time student calculations) and refuses to extrapolate or hallucinate unwritten rules.
- **Zero-Embedding BM25 Retrieval**: Fast, deterministic lexical indexing with domain-specific synonym expansion.
- **Comprehensive 12-Question Evaluation Suite**: Automated, reproducible evaluation covering table lookups, disregards, resource limits, needs figures, paraphrased queries, edge cases, contradictions, gaps, and refusals.
- **Offline / Dry-Run Mode**: Full test suite and evaluation can run without an API key.

---

## Architecture

```text
                  User Policy Question
                           │
                           ▼
               1. Query Expansion & BM25 Retrieval
                  (Clause-Level Granularity)
                           │
                           ▼
               2. Evidence Validation & Context Construction
                  (Top-k clauses, strict grounding prompt)
                           │
                           ▼
               3. LLM Generation (Groq / gpt-oss-20b)
                  (Zero hallucination, strict citation rules)
                           │
                           ▼
               4. Decision, Status & Citation Extraction
                  ┌───────────────┬─────────────────┐
                  ▼               ▼                 ▼
          Answered (+ §IDs)    Contradiction      Refusal / Gap
                               (Dual §IDs)        (+ Next Step)
```

---

## Requirements

- **Python**: Version `3.11` or later (tested on Python `3.13`)
- **API Key**: A [Groq Cloud API Key](https://console.groq.com/) (`GROQ_API_KEY`) for live model generation (optional for dry-run/unit tests).
- **Core Dependencies**:
  - `rank-bm25` (BM25Okapi retrieval)
  - `openai` (OpenAI-compatible client for Groq inference)
  - `python-dotenv` (Environment variable management)
  - `pytest` (Automated testing framework)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd grounded-answer
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On Windows (PowerShell):
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # On Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

1. Copy the example environment file:
   ```bash
   # On Windows (PowerShell):
   Copy-Item .env.example .env

   # On Linux / macOS:
   cp .env.example .env
   ```

2. Open `.env` and configure your credentials:
   ```ini
   GROQ_API_KEY=gsk_your_actual_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```

> **Note**: The `.env` file is in `.gitignore` and is never committed.

---

## Running the Application

### 1. Ask a Single Question via CLI
```bash
python src/main.py --question "What is the monthly income threshold for a household of 4?"
```

### 2. Interactive CLI Mode
```bash
python src/main.py
```
*(Type questions interactively and type `q` or `quit` to exit)*

### 3. Dry-Run Mode (No API Key Required)
Inspect retrieved policy clauses without calling the LLM:
```bash
python src/main.py --dry-run --question "How many days does a recipient have to report a change of circumstances?"
```

### 4. JSON Output
```bash
python src/main.py --json --question "What is the monthly needs figure for a couple?"
```

---

## Example Usage and Expected Outputs

### Example 1: Normal Answerable Question
```bash
python src/main.py --question "What is the monthly income threshold for a household of 4?"
```
**Output:**
```text
Status: [ANSWERED]
Citations: §6.6.1
Answer:
The monthly income threshold for a household of 4 is $2,410.
§6.6.1.
```

### Example 2: Genuine Contradiction Handling
```bash
python src/main.py --question "How many days does a recipient have to report a change of circumstances to the Department?"
```
**Output:**
```text
Status: [CONTRADICTION]
Citations: §4.3.2, §9.1.4
Answer:
The policy manual contains conflicting provisions.
Clause §4.3.2 states that a recipient must report a change of circumstances within 10 calendar days.
Clause §9.1.4 states that no overpayment shall be established if the recipient reported the change within the 30 calendar days required under §4.3.
These provisions cannot both be applied and the manual does not provide a consistent basis to determine the outcome.
Next step: Escalate to the policy administrator/program authority for clarification.
```

### Example 3: Apparent Policy Gap / Broken Reference
```bash
python src/main.py --question "How is the monthly needs figure calculated for a household member who is a full-time student?"
```
**Output:**
```text
Status: [GAP]
Citations: §7.1.3, §5.4
Answer:
Unable to determine from the policy manual.
Clause §7.1.3 states that the needs figure is calculated by reference to household size and composition 'except in the case of full-time students (see §5.4)', but §5.4 only addresses care allowances and does not specify a calculation rule for full-time students.
Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification.
```

### Example 4: Out-of-Scope Policy Refusal
```bash
python src/main.py --question "What dental and optical care expenses are covered under the Household Support Program?"
```
**Output:**
```text
Status: [REFUSAL]
Answer:
Unable to determine from the policy manual.
The supplied evidence does not contain information on dental or optical care coverage.
Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification.
```

---

## Evaluation

Run the dedicated 12-question evaluation suite:

### Live Evaluation (with Groq API)
```bash
python evaluate.py
```

### Dry-Run Evaluation (Retrieval verification, no API key needed)
```bash
python evaluate.py --dry-run
```

### Raw JSON Evaluation Report
```bash
python evaluate.py --json
```

---

## Automated Software Tests

Run the full automated test suite (62+ tests covering chunking, retrieval, prompt formatting, pipeline orchestration, mocking, and edge cases):

```bash
python -m pytest -v
```

---

## Project Structure

```text
grounded-answer/
├── data/
│   ├── policy-manual.md     # Consolidated Calder County HSP policy manual (12 Parts)
│   └── README.md            # Data pack documentation
├── src/
│   ├── chunker.py           # Clause-level Markdown parser (§Part.Section.Paragraph)
│   ├── retriever.py         # BM25Okapi retriever with domain query expansion
│   ├── generator.py         # Grounded answer synthesis & system prompts
│   ├── pipeline.py          # Decoupled end-to-end RAG pipeline
│   ├── evaluator.py         # 12-question evaluation dataset and evaluation harness
│   └── main.py              # CLI entry point
├── tests/
│   ├── test_retriever.py    # Unit tests for tokenization and BM25 retrieval
│   ├── test_generator.py    # Pure unit tests for prompt building & mock generation
│   ├── test_pipeline.py     # Integration tests for pipeline stages & error handling
│   └── test_evaluation.py   # Validation tests for evaluation dataset & dry run
├── evaluate.py              # CLI evaluation runner producing formatted Markdown tables
├── DECISIONS.md             # Architectural & engineering decisions document
├── AIUSAGE.md               # AI assistance disclosure and human review record
├── requirements.txt         # Project dependencies
├── .env.example             # Template environment variables
└── README.md                # Project documentation
```
