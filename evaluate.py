#!/usr/bin/env python3
"""
evaluate.py — Dedicated evaluation runner for Brite Spark 2026 Problem 1.

Runs the 12-question evaluation suite against the Grounded Answer RAG pipeline,
checks groundedness, citations, contradiction handling, apparent-gap handling,
refusal paths, and outputs a formatted Markdown/console summary table with
honest PASS/FAIL results.

Usage:
    python evaluate.py
    python evaluate.py --dry-run
    python evaluate.py --top-k 5
    python evaluate.py --json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import GroundedAnswerPipeline
from src.evaluator import EVALUATION_DATASET, run_all_evaluations


# Ensure stdout handles UTF-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_markdown_table(results: list[dict]) -> None:
    print("\n# Evaluation Report — Calder County Household Support Program\n")
    print("| ID | Category | Question | Expected | Actual | Citations | Result |")
    print("|:---|:---|:---|:---|:---|:---|:---|")

    passed_count = 0
    failed_count = 0

    for r in results:
        res_str = "**PASS**" if r["passed"] else "**FAIL**"
        if r["passed"]:
            passed_count += 1
        else:
            failed_count += 1

        citations_str = ", ".join(r["citations"]) if r["citations"] else "-"
        # Truncate question if necessary
        q_short = r["question"] if len(r["question"]) <= 50 else r["question"][:47] + "..."

        print(
            f"| {r['id']} | {r['category']} | {q_short} | "
            f"{r['expected_status']} | {r['actual_status']} | "
            f"{citations_str} | {res_str} |"
        )

    print("\n## Evaluation Summary")
    print(f"- **Total Questions**: {len(results)}")
    print(f"- **Passed**: {passed_count}")
    print(f"- **Failed**: {failed_count}")
    print(f"- **Pass Rate**: {(passed_count / len(results)) * 100:.1f}%\n")

    # Detailed notes on special cases
    print("## Detailed Question Results\n")
    for r in results:
        status_symbol = "PASS" if r["passed"] else "FAIL"
        print(f"### [{status_symbol}] {r['id']}: {r['question']}")
        print(f"- **Category**: {r['category']}")
        print(f"- **Expected Status**: `{r['expected_status']}` | **Actual Status**: `{r['actual_status']}`")
        print(f"- **Evaluation Result**: {r['reason']}")
        if r["citations"]:
            print(f"- **Citations Found**: {', '.join(r['citations'])}")
        print(f"- **Pipeline Answer**:\n  > {r['answer'].replace(chr(10), chr(10) + '  > ')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 12-question evaluation suite for Calder County HSP RAG."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run retrieval-only evaluation without LLM calls.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of policy clauses to retrieve (default: 5).",
    )
    parser.add_argument(
        "--manual",
        help="Path to the policy manual Markdown file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results.",
    )
    args = parser.parse_args()

    dry_run = args.dry_run or not os.environ.get("GROQ_API_KEY")

    if dry_run and not args.dry_run:
        print("[INFO] No GROQ_API_KEY detected in environment. Running in dry-run mode.")

    try:
        pipeline = GroundedAnswerPipeline.build(manual_path=args.manual)
    except Exception as exc:
        print(f"Error initializing pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    results = run_all_evaluations(pipeline, top_k=args.top_k, dry_run=dry_run)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_markdown_table(results)

    # Exit code: 0 if all pass, 1 if failures
    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
