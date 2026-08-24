# Architectural and Design Decisions — Grounded Answer (Calder County HSP)

This document records the architectural, algorithmic, and policy decisions made during the design and implementation of the Grounded Answer RAG system for the Calder County Household Support Program (HSP).

---

## 1. Retrieval Strategy

### Choice: BM25Okapi with Policy-Domain Query Expansion

#### Rationale
- **Lexical Precision for Statutory/Policy Text**: Benefits administration manuals are structured around strict numeric thresholds (`$2,410`), temporal limits (`10 calendar days`, `28 days`, `90 days`), percentages (`20 per cent`), and explicit clause cross-references (`§6.6.1`, `§4.3.2`). BM25 provides exact keyword and numerical matching without embedding dilution.
- **Zero Embedding Drift & Zero External Vector Dependency**: Dense vector models often hallucinate semantic proximity between unrelated numbers or map distinct policy sections together (e.g., confusing the $4,000 resource limit in §2.4.1 with the $2,410 income threshold in §6.6.1). BM25 prevents numerical and clause drift.
- **Fast Cold-Start & Deterministic Execution**: The index is constructed in under 50ms upon pipeline initialization from the local Markdown corpus, requiring no external vector database, embedding API keys, or GPU compute.
- **Targeted Query Expansion**: Natural language queries frequently use colloquial terms ("away from county", "car", "salary", "report deadline", "minimum award") while the manual uses formal administrative terminology ("absence", "motor vehicle", "earnings", "change of circumstances", "resulting figure less than $25"). Lightweight synonym expansion bridges vocabulary mismatches without altering the user's intent.

---

## 2. Document Chunking Strategy

### Choice: Clause-Level Semantic Paragraph Granularity

#### Granularity
- Every clause matching the numbered pattern `§Part.Section.Paragraph` (e.g., `§4.3.2`) is parsed as an atomic unit.
- Sub-items (`(a)`, `(b)`, `(c)`), condition lists, and inline Markdown tables (e.g., the income threshold schedule in `§6.6.1` and needs figures in `§7.2.1`) are kept intact within their parent clause.
- Each clause dictionary preserves:
  - `id`: Stable clause citation (e.g., `"§4.3.2"`).
  - `part`: Integer Part number (1 to 12).
  - `section`: Section identifier (e.g., `"4.3"`).
  - `heading`: Immediate section header (e.g., `"Recipient obligations"`).
  - `text`: Complete textual content of the clause.

#### Why not fixed-size sliding windows?
Fixed-token sliding windows (e.g., 256 tokens with 50-token overlap) split sub-clauses from their parent rules, separate table headers from their row values, and sever clause numbers from the text. Clause-level chunking aligns 1-to-1 with the manual's legal structure and caseworker citation practices.

---

## 3. Answer / Refusal Boundary and Calibration

### Policy Context & Risk Calibration
In public assistance programs, **generating an incorrect or ungrounded answer carries severe consequences**:
- Unlawful entitlement awards or wrongful claim denials.
- Recipient hardship and subsequent administrative overpayment recovery.
- Administrative audit sanctions against the Department.

Therefore, the system is intentionally calibrated to be **conservative**: it is far better to issue a structured refusal with a clear next step than to produce a fluent, plausible guess.

### Refusal Signals & Multi-Factor Boundary
A refusal is triggered under any of the following conditions:
1. **Empty Query or No Evidence Retrieved**: When BM25 returns zero clauses or the query is blank, the pipeline returns the standard insufficient-evidence refusal without invoking the LLM.
2. **Out-of-Scope Topics**: If the query concerns subjects absent from the manual (e.g., optical/dental coverage, business startup grants), the model strictly adheres to prompt grounding and refuses to extrapolate from general knowledge.
3. **Apparent Policy Gaps / Broken References**: If a topic is mentioned in the manual (e.g., full-time students in §1.4.6 and §7.1.3) but the governing operative calculation rule is missing, the system refuses to guess the calculation.
4. **Internal Contradictions**: When two provisions directly conflict (§4.3.2 vs §9.1.4), the system refuses to make an unauthorized policy choice.

### Actionable Next Steps
Every refusal provides an operational directive tailored to the caseworker:
```text
Unable to determine from the policy manual.
[Reason for refusal / missing policy rule]
Next step: Please consult the designated policy/program authority or caseworker supervisor for clarification.
```

---

## 4. Contradiction Handling

### Corpus Finding: Reporting Deadlines (§4.3.2 vs §9.1.4)
The policy manual contains a genuine internal contradiction regarding the change of circumstances reporting timeframe:
- **§4.3.2 (Recipient obligations)**: *"A recipient must report any change in household composition, income, address, or the circumstances of any household member within **10 calendar days** of the change occurring..."*
- **§9.1.4 (Establishing an overpayment)**: *"Where an overpayment has arisen from a change of circumstances, and the recipient reported the change within the **30 calendar days** required under §4.3, no overpayment shall be established..."*

### Architectural Response
1. **Dual Retrieval**: Both §4.3.2 and §9.1.4 are retrieved in the top evidence set for change-of-circumstances reporting questions.
2. **Neutral Presentation**: The prompt and post-processing ensure the LLM does not arbitrarily pick 10 days or 30 days.
3. **Explicit Conflict Declaration**: The system outputs:
   - "The policy manual contains conflicting provisions."
   - Displays both Clause §4.3.2 (10 days) and Clause §9.1.4 (30 days).
   - Explains that the manual does not provide a consistent basis to determine which deadline governs.
   - Directs the caseworker: *"Next step: Escalate to the policy administrator/program authority for clarification."*

---

## 5. Apparent Policy Gap Handling

### Corpus Finding: Full-Time Student Needs Calculation (§7.1.3 vs §5.4)
- **§7.1.3 (The award)** states: *"The needs figure is calculated by reference to household size and composition, **except in the case of full-time students (see §5.4)**, and is subject to the adjustments in §7.3."*
- **§5.4** is titled *"Households including a person in receipt of a care allowance"* and only governs care allowances. It contains zero rules or formulas for full-time students.
- Furthermore, **§3.2.3** and **§5.2.3** note that *"Temporary absence does not include absence for the purpose of full-time education, which is addressed separately"*, but no such separate section exists.

### Architectural Response
Naive semantic search or unconstrained LLMs would attempt to synthesize an answer from general student assistance knowledge or confuse the care allowance in §5.4 with a student allowance. 

Our prompt instructions explicitly forbid completing broken cross-references. The system recognizes that while the category is defined (§1.4.6) and referenced (§7.1.3), the substantive operative rule is absent, triggering a structured refusal.

---

## 6. Citation Design

### Verifiability and Traceability
- All citations adhere strictly to the manual's numbering scheme: `§Part.Section.Paragraph` (e.g., `§6.6.1`).
- Every substantive claim in an answer is followed by or linked to its specific clause ID.
- The pipeline extracts unique citations from the generated answer and validates them against the retrieved evidence to ensure no fabricated clause numbers exist.

---

## 7. Allocation of LLM vs Deterministic Logic

| Function | Implementation | Rationale |
|:---|:---|:---|
| Corpus Parsing & Chunking | Deterministic (Regex) | Precise boundary detection on Markdown headers |
| Retrieval Indexing & Scoring | Deterministic (BM25Okapi) | Exact keyword/numeric match, reproducible |
| Query Expansion | Deterministic (Regex Rules) | Controlled domain vocabulary mapping |
| Evidence Synthesis & Grounding | LLM (Groq / gpt-oss-20b) | Natural language synthesis strictly constrained to supplied evidence |
| Citation Extraction | Deterministic (Regex) | Robust identification of `§X.Y.Z` tokens |
| Decision / Status Classification | Deterministic (String & Rule matching) | Objective, reproducible PASS/FAIL categorization |
| Evaluation Scoring | Deterministic Harness | Automated verification of statuses, citations, and key terms |

---

## 8. Day-Two Requirement Change Adaptability

The system separates the pipeline into distinct, independently testable phases:

```text
Question
   │
   ▼
1. Retrieval (retrieve_evidence)
   │
   ▼
2. Evidence Validation & Context Construction (build_prompt)
   │
   ▼
3. Grounded Synthesis (generate_answer)
   │
   ▼
4. Decision & Status Evaluation (evaluate_decision / classify_status)
   │
   ▼
Structured Output (answer, evidence, citations, status, is_refusal)
```

Each stage is decoupled in `GroundedAnswerPipeline`. Should Day-Two introduce changes (e.g., alternative retrieval algorithms, custom citation formatting, modified threshold boundaries, or different model backends), individual components can be updated or swapped without refactoring the rest of the application.

---

## 9. Day-Two Adaptation Record: Amendment No. 2026-01 & Temporal Grounding

### Context of the Requirement Change
On Day Two, **Amendment No. 2026-01** (issued 12 February 2026, effective 1 March 2026) was added to the corpus. The system was required to provide answers that are correct for the **specific date of the claim being asked about**, rather than assuming a single static state of policy.

### Summary of Policy Changes in Amendment No. 2026-01
1. **Earnings Disregard (§6.4.1(a))**: Increased from **$120/month** to **$175/month** (effective 1 March 2026).
2. **Reporting Deadlines (§4.3.2 & §9.1.4)**: Aligned both clauses to **14 calendar days** for changes occurring on or after 1 March 2026 (resolving the prior 10-day vs 30-day contradiction).
3. **Income Thresholds (§6.6.1)**: Updated schedule for determinations on/after 1 March 2026 (e.g., Household of 4 increased from **$2,410** to **$2,500**).
4. **Sanctions (§10.5.2 & §10.5.3A)**: Standard sanction rate reduced from **20%** to **15%**, and a new exemption (**§10.5.3A**) was enacted prohibiting sanctions for failure to report where the change would have increased the award.
5. **Transitional Provisions (¶5.1 - ¶5.3)**: 
   - Paragraphs 1, 3, and 4 apply to determinations made on or after 1 March 2026.
   - Paragraph 2 applies to changes of circumstances occurring on or after 1 March 2026 (pre-March 2026 changes remain subject to the old rules/contradiction).

### What Was Changed
- **Corpus Ingestion (`src/chunker.py`)**: Enhanced `load_policy()` to automatically discover and parse amendment files alongside `policy-manual.md`. Amendment paragraphs are chunked with distinct identifiers (`Amendment 2026-01 ¶X.Y`) and tagged with `effective_date: 1 March 2026`.
- **Query Expansion (`src/retriever.py`)**: Added terms for new numerical thresholds ($175, $2,500, 15%, 14 days, §10.5.3A) and temporal date markers ("February 2026", "April 2026", "March 2026", "amendment", "effective").
- **Temporal Prompt Framing (`src/generator.py`)**: Instructed the LLM to inspect the date of the claim/determination. Pre-1 March 2026 claims are answered using base manual rules; post-1 March 2026 claims are answered using Amendment No. 2026-01.
- **Citation Extraction (`src/pipeline.py`)**: Expanded citation regex to support lettered clauses (`§10.5.3A`) and amendment citations (`Amendment 2026-01 ¶3.1`).
- **Evaluation Suite (`src/evaluator.py`)**: Expanded to 17 questions specifically verifying temporal pairs (Feb 2026 vs April 2026 claims across thresholds, disregards, sanctions, and reporting deadlines).

### What Was Chosen NOT to Change
- **BM25 Retrieval Engine**: We chose not to replace BM25 with a complex vector store or time-indexed database. Feeding both the base manual clause and the corresponding amendment paragraph in the top-5 evidence set allows the model to perform exact temporal reasoning without embedding hallucination.
- **Pipeline Abstraction**: The 4-stage pipeline structure (`retrieve_evidence` -> `generate_answer` -> `extract_citations` -> `evaluate_decision`) established on Day 1 absorbed the amendment without any breaking architectural redesign.

### What We Would Have Done Differently if Known Upfront
- If temporal versioning and multi-amendment lifecycles had been specified in the original requirements, we would have implemented a formal clause schema with explicit date intervals:
  ```json
  {
    "clause_id": "§6.4.1(a)",
    "valid_from": "2025-12-31",
    "valid_to": "2026-02-28",
    "disregard": 120
  }
  ```
  However, the dual-corpus retrieval approach combined with LLM temporal instruction proved exceptionally resilient, modular, and easy to maintain.

