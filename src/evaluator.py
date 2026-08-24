"""
evaluator.py — 12-question evaluation harness for Calder County HSP RAG.

Evaluates:
- Normal answerable questions (table lookup, disregards, adjustments)
- Citation-accuracy questions
- Paraphrased natural-language questions
- Edge cases and exceptions
- Genuine internal contradiction (§4.3.2 vs §9.1.4)
- Apparent policy gap / broken reference (§7.1.3 -> §5.4 full-time students)
- Out-of-scope refusals
"""

import re
from typing import Any
from src.pipeline import GroundedAnswerPipeline

EVALUATION_DATASET: list[dict[str, Any]] = [
    {
        "id": "Q01",
        "category": "Temporal / Pre-Amendment Threshold",
        "question": "For a claim dated February 2026, what is the monthly income threshold for a household of 4?",
        "expected_status": "answered",
        "expected_clauses": ["§6.6.1"],
        "key_terms": ["2,410", "$2410", "$2,410"],
        "description": "Tests pre-amendment income threshold table lookup (§6.6.1).",
    },
    {
        "id": "Q02",
        "category": "Temporal / Post-Amendment Threshold",
        "question": "For a claim dated April 2026, what is the monthly income threshold for a household of 4?",
        "expected_status": "answered",
        "expected_clauses": ["§6.6.1"],
        "key_terms": ["2,500", "$2500", "$2,500"],
        "description": "Tests post-amendment income threshold under Amendment 2026-01 ¶3.1.",
    },
    {
        "id": "Q03",
        "category": "Temporal / Pre-Amendment Disregard",
        "question": "For a determination made in January 2026, how much monthly employment earnings are disregarded per household?",
        "expected_status": "answered",
        "expected_clauses": ["§6.4.1"],
        "key_terms": ["120", "$120"],
        "description": "Tests pre-amendment earnings disregard rule ($120) in §6.4.1(a).",
    },
    {
        "id": "Q04",
        "category": "Temporal / Post-Amendment Disregard",
        "question": "For a determination made in April 2026, how much monthly employment earnings are disregarded per household?",
        "expected_status": "answered",
        "expected_clauses": ["§6.4.1"],
        "key_terms": ["175", "$175"],
        "description": "Tests post-amendment earnings disregard rule ($175) under Amendment 2026-01 ¶1.1.",
    },
    {
        "id": "Q05",
        "category": "Citation / Resource Limit",
        "question": "What is the maximum countable resource limit and is a motor vehicle excluded?",
        "expected_status": "answered",
        "expected_clauses": ["§2.4.1", "§2.4.2"],
        "key_terms": ["4,000", "$4,000", "motor vehicle", "vehicle", "excluded", "not countable"],
        "description": "Tests countable resources limit and primary motor vehicle exclusion.",
    },
    {
        "id": "Q06",
        "category": "Citation / Needs Schedule",
        "question": "What is the monthly needs figure for a couple?",
        "expected_status": "answered",
        "expected_clauses": ["§7.2.1"],
        "key_terms": ["1,670", "$1,670", "$1670"],
        "description": "Tests baseline needs figure lookup for couple composition.",
    },
    {
        "id": "Q07",
        "category": "Answerable (Adjustments)",
        "question": "By how much is the needs figure increased if a household member requires assistance with two or more activities of daily living?",
        "expected_status": "answered",
        "expected_clauses": ["§7.3.1"],
        "key_terms": ["90", "$90"],
        "description": "Tests specific needs adjustment for ADL assistance in §7.3.1.",
    },
    {
        "id": "Q08",
        "category": "Paraphrased Query",
        "question": "How long can a beneficiary stay out of Calder County to receive specialized medical treatment without losing their eligibility?",
        "expected_status": "answered",
        "expected_clauses": ["§3.2.2"],
        "key_terms": ["90 days", "90", "medical"],
        "description": "Tests paraphrased query on temporary medical absence exception (§3.2.2(a)).",
    },
    {
        "id": "Q09",
        "category": "Edge Case / Sanction Exception",
        "question": "Can the Department impose a sanction on a household that includes a 1-year-old child?",
        "expected_status": "answered",
        "expected_clauses": ["§10.5.3"],
        "key_terms": ["must not", "no", "cannot", "prohibited", "under the age of 2", "under 2", "child"],
        "description": "Tests statutory exemption from sanctions for households with child under 2.",
    },
    {
        "id": "Q10",
        "category": "Temporal / Post-Amendment Sanction Rate",
        "question": "For a determination made in April 2026, what percentage reduction applies to a first sanction?",
        "expected_status": "answered",
        "expected_clauses": ["§10.5.2"],
        "key_terms": ["15", "15 per cent", "15%"],
        "description": "Tests post-amendment reduced sanction rate (15%) under Amendment 2026-01 ¶4.1.",
    },
    {
        "id": "Q11",
        "category": "Temporal / New Sanction Exemption",
        "question": "Under Amendment No. 2026-01, can a sanction be imposed for a failure to report if the change would have increased the award?",
        "expected_status": "answered",
        "expected_clauses": ["§10.5.3A"],
        "key_terms": ["must not", "no", "cannot", "prohibited", "increased"],
        "description": "Tests new §10.5.3A sanction exemption inserted by Amendment 2026-01 ¶4.2.",
    },
    {
        "id": "Q12",
        "category": "Edge Case / Minimum Award",
        "question": "Will an award be paid if the calculated monthly entitlement after deducting income comes out to $18?",
        "expected_status": "answered",
        "expected_clauses": ["§7.1.2"],
        "key_terms": ["no award", "less than $25", "less than 25", "$25", "not made", "will not"],
        "description": "Tests minimum award threshold rule ($25 floor).",
    },
    {
        "id": "Q13",
        "category": "Pre-Amendment Contradiction",
        "question": "For a change of circumstances occurring in January 2026, how many days does a recipient have to report the change?",
        "expected_status": "contradiction",
        "expected_clauses": ["§4.3.2", "§9.1.4"],
        "key_terms": ["conflict", "conflicting", "10", "30", "escalat", "administrator"],
        "description": "Tests pre-amendment contradiction between §4.3.2 (10 days) and §9.1.4 (30 days).",
    },
    {
        "id": "Q14",
        "category": "Temporal / Post-Amendment Reporting Deadline",
        "question": "For a change of circumstances occurring in April 2026, how many calendar days does a recipient have to report the change?",
        "expected_status": "answered",
        "expected_clauses": ["§4.3.2"],
        "key_terms": ["14", "14 calendar days", "14 days"],
        "description": "Tests aligned 14-day reporting deadline under Amendment 2026-01 ¶2.1 and ¶2.2.",
    },
    {
        "id": "Q15",
        "category": "Apparent Policy Gap",
        "question": "How is the monthly needs figure calculated for a household member who is a full-time student?",
        "expected_status": "gap",
        "expected_clauses": ["§7.1.3", "§5.4"],
        "key_terms": ["unable to determine", "insufficient", "does not specify", "not provided", "gap", "next step"],
        "description": "Tests apparent policy gap / broken reference in §7.1.3 pointing to §5.4.",
    },
    {
        "id": "Q16",
        "category": "Refusal (Out of Scope)",
        "question": "What dental and optical care expenses are covered under the Household Support Program?",
        "expected_status": "refusal",
        "expected_clauses": [],
        "key_terms": ["unable to determine", "insufficient", "not covered", "next step"],
        "description": "Tests refusal when topic is completely absent from the manual.",
    },
    {
        "id": "Q17",
        "category": "Refusal (Unsupported Inference)",
        "question": "Can an applicant receive a grant under this program to fund business startup costs?",
        "expected_status": "refusal",
        "expected_clauses": [],
        "key_terms": ["unable to determine", "insufficient", "does not", "next step"],
        "description": "Tests refusal for unauthorized policy purpose / business startup funding.",
    },
]


def evaluate_question(
    pipeline: GroundedAnswerPipeline,
    case: dict[str, Any],
    top_k: int = 5,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run evaluation on a single question case and determine PASS/FAIL.
    """
    res = pipeline.ask(case["question"], top_k=top_k, dry_run=dry_run)
    answer = res["answer"]
    status = res["status"]
    citations = res["citations"]
    evidence_ids = [c["id"] for c in res["evidence"]]

    passed = True
    reasons = []

    if dry_run:
        # In dry run, check whether required clauses are in retrieved evidence
        if case["expected_status"] == "gap":
            if not any(any(req in eid for eid in evidence_ids) for req in case["expected_clauses"]):
                passed = False
                reasons.append(f"Missing required gap clause ({case['expected_clauses']}) in retrieved evidence")
        else:
            for req_clause in case["expected_clauses"]:
                if not any(req_clause in eid for eid in evidence_ids):
                    passed = False
                    reasons.append(f"Missing required clause {req_clause} in retrieved evidence")
        return {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_status": case["expected_status"],
            "actual_status": status,
            "passed": passed,
            "reason": "; ".join(reasons) if reasons else "Dry-run retrieval verified",
            "citations": citations,
            "evidence_ids": evidence_ids,
            "answer": answer,
        }

    # 1. Status verification
    if case["expected_status"] == "contradiction":
        if status != "contradiction":
            # Check if answer mentions both conflicting clauses and escalation
            if "§4.3.2" in answer and "§9.1.4" in answer and ("conflict" in answer.lower() or "inconsisten" in answer.lower()):
                pass
            else:
                passed = False
                reasons.append(f"Expected contradiction status, got '{status}'")
    elif case["expected_status"] in ("refusal", "gap"):
        if status not in ("refusal", "gap"):
            passed = False
            reasons.append(f"Expected refusal/gap status, got '{status}'")
    elif case["expected_status"] == "answered":
        if status != "answered":
            passed = False
            reasons.append(f"Expected answered status, got '{status}'")

    # 2. Citation verification for answered, contradiction, and gap cases
    normalized_answer = re.sub(r"§[\s\u2000-\u200b\u202f\u00a0]*", "§", answer)
    if case["expected_status"] == "gap":
        if case["expected_clauses"] and not any(c in citations or c in normalized_answer for c in case["expected_clauses"]):
            passed = False
            reasons.append(f"Missing required gap citation among {case['expected_clauses']}")
    else:
        for req_clause in case["expected_clauses"]:
            # Check in either parsed citations or normalized answer
            if req_clause not in citations and req_clause not in normalized_answer:
                passed = False
                reasons.append(f"Missing required citation {req_clause}")

    # 3. Next step verification for refusal/contradiction/gap
    if case["expected_status"] in ("refusal", "gap", "contradiction"):
        if "next step" not in answer.lower():
            passed = False
            reasons.append("Missing actionable 'Next step:' instruction")

    # 4. Key terms check
    if case.get("key_terms"):
        matched_any = any(term.lower() in answer.lower() for term in case["key_terms"])
        if not matched_any:
            passed = False
            reasons.append(f"Answer did not contain expected substantive terms: {case['key_terms']}")

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_status": case["expected_status"],
        "actual_status": status,
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else "Output satisfied all grounded policy criteria",
        "citations": citations,
        "evidence_ids": evidence_ids,
        "answer": answer,
    }


def run_all_evaluations(
    pipeline: GroundedAnswerPipeline,
    top_k: int = 5,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run evaluation for all questions in the dataset."""
    results = []
    for case in EVALUATION_DATASET:
        res = evaluate_question(pipeline, case, top_k=top_k, dry_run=dry_run)
        results.append(res)
    return results
