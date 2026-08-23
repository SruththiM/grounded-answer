"""
main.py — CLI entry point for the grounded-answer pipeline.

Usage:
    python src/main.py
    python src/main.py --question "What is the income threshold for a household of 3?"
    python src/main.py --dry-run

Environment variables:
    GROQ_API_KEY  — required for live answers
    GROQ_MODEL    — optional, defaults to gpt-4o-mini
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running as  python src/main.py  without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import GroundedAnswerPipeline
from src.generator import _NO_EVIDENCE_ANSWER


def _print_result(question: str, result: dict, dry_run: bool = False) -> None:
    print(f"\nQuestion: {question}")
    print("-" * 60)

    evidence = result["evidence"]
    if evidence:
        print("Retrieved policy evidence:")
        for c in evidence:
            print(f"  {c['id']}  [{c['heading']}]")
    else:
        print("Retrieved policy evidence: (none)")

    print()
    if dry_run:
        print("(dry-run — LLM call skipped)")
    else:
        print(f"Answer:\n{result['answer']}")
    print()


def _interactive_loop(pipeline: GroundedAnswerPipeline, dry_run: bool) -> None:
    print("Calder County Household Support Program — Policy Assistant")
    print("Type your question and press Enter.  Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        try:
            result = pipeline.ask(question)
        except EnvironmentError as exc:
            print(f"\nConfiguration error: {exc}\n")
            continue

        _print_result(question, result, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grounded policy Q&A — Calder County Household Support Program"
    )
    parser.add_argument(
        "--question", "-q",
        help="Ask a single question and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show retrieved evidence without calling the LLM.",
    )
    parser.add_argument(
        "--manual",
        help="Path to the policy manual Markdown file (default: data/policy-manual.md).",
    )
    args = parser.parse_args()

    dry_run = args.dry_run or not os.environ.get("GROQ_API_KEY")

    try:
        pipeline = GroundedAnswerPipeline.build(manual_path=args.manual)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading policy manual: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        try:
            result = pipeline.ask(args.question)
        except EnvironmentError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            sys.exit(1)
        _print_result(args.question, result, dry_run=dry_run)
    else:
        _interactive_loop(pipeline, dry_run=dry_run)


if __name__ == "__main__":
    main()
