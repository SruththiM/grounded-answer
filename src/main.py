"""
main.py — CLI entry point for the grounded-answer pipeline.

Usage:
    python src/main.py
    python src/main.py --question "What is the income threshold for a household of 3?"
    python src/main.py --dry-run

Environment variables:
    GROQ_API_KEY  — required for live answers
    GROQ_MODEL    — optional, defaults to openai/gpt-oss-20b
"""
import argparse
import json
import os
import sys
from pathlib import Path
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# Allow running as  python src/main.py  without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import GroundedAnswerPipeline


def _print_result(question: str, result: dict, dry_run: bool = False) -> None:
    print(f"\nQuestion: {question}")
    print("=" * 65)

    status = result.get("status", "answered").upper()
    print(f"Status: [{status}]")

    evidence = result.get("evidence", [])
    if evidence:
        print("\nRetrieved Policy Evidence:")
        for c in evidence:
            score_str = f"(score: {c.get('score', 0):.2f})" if "score" in c else ""
            print(f"  • {c['id']} — {c['heading']} {score_str}")
    else:
        print("\nRetrieved Policy Evidence: (none)")

    citations = result.get("citations", [])
    if citations:
        print(f"\nCitations: {', '.join(citations)}")

    print("-" * 65)
    if dry_run:
        print("(dry-run — LLM call skipped)")
    else:
        print(f"Answer:\n{result['answer']}")
    print("=" * 65 + "\n")


def _interactive_loop(pipeline: GroundedAnswerPipeline, dry_run: bool, top_k: int = 5) -> None:
    print("=================================================================")
    print(" Calder County Household Support Program — Policy Assistant")
    print("=================================================================")
    print("Type your question and press Enter.  Type 'quit' or 'q' to exit.\n")

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
            result = pipeline.ask(question, top_k=top_k, dry_run=dry_run)
        except EnvironmentError as exc:
            print(f"\nConfiguration error: {exc}\n")
            continue

        _print_result(question, result, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grounded Policy Q&A — Calder County Household Support Program"
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
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of policy clauses to retrieve (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format.",
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
            result = pipeline.ask(args.question, top_k=args.top_k, dry_run=dry_run)
        except EnvironmentError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_result(args.question, result, dry_run=dry_run)
    else:
        _interactive_loop(pipeline, dry_run=dry_run, top_k=args.top_k)


if __name__ == "__main__":
    main()

