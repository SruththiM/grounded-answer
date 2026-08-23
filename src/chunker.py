"""
chunker.py — splits policy-manual.md into clause-level dicts.

Each dict has:
    id      : "§4.3.2"
    part    : 4
    section : "4.3"
    heading : "Recipient obligations"   (nearest ## heading)
    text    : full paragraph text including sub-items and tables
"""

import re
from pathlib import Path

# Matches the opening of a numbered clause, e.g. **4.3.2** or **1.4.1 Applicant**
_CLAUSE_START = re.compile(r"^\*\*(\d+)\.(\d+)\.(\d+)(?:[^*]*)?\*\*")

# Matches a ## section heading, e.g. ## 4.3 Recipient obligations
_SECTION_HEADING = re.compile(r"^## (\d+\.\d+)\s+(.*)")


def load_policy(path: str) -> list[dict]:
    """
    Parse the policy manual at *path* and return a list of clause dicts.
    The file is read as UTF-8.  The source file is never modified.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    clauses: list[dict] = []
    current_heading = ""
    current_clause: dict | None = None
    current_lines: list[str] = []

    def _flush():
        """Save the accumulated lines into current_clause and append it."""
        if current_clause is None:
            return
        current_clause["text"] = "\n".join(current_lines).strip()
        clauses.append(current_clause)

    for line in lines:
        # Track the nearest ## section heading
        heading_match = _SECTION_HEADING.match(line)
        if heading_match:
            current_heading = heading_match.group(2).strip()
            continue

        # Detect a new clause boundary
        clause_match = _CLAUSE_START.match(line)
        if clause_match:
            _flush()
            part_num = int(clause_match.group(1))
            section_str = f"{clause_match.group(1)}.{clause_match.group(2)}"
            para_str = f"{section_str}.{clause_match.group(3)}"
            current_clause = {
                "id": f"§{para_str}",
                "part": part_num,
                "section": section_str,
                "heading": current_heading,
            }
            current_lines = [line]
            continue

        # Accumulate into the current clause (sub-items, tables, blank lines)
        if current_clause is not None:
            current_lines.append(line)

    # Flush the final clause
    _flush()

    return clauses


# ---------------------------------------------------------------------------
# Inspection helper — run:  python src/chunker.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    manual_path = Path(__file__).parent.parent / "data" / "policy-manual.md"
    clauses = load_policy(str(manual_path))

    print(f"Total clauses : {len(clauses)}")
    print()

    print("First 3 clause IDs:")
    for c in clauses[:3]:
        print(f"  {c['id']}  ({c['heading']})")
    print()

    print("Last 3 clause IDs:")
    for c in clauses[-3:]:
        print(f"  {c['id']}  ({c['heading']})")
    print()

    # Table checks
    for clause_id in ("§6.6.1", "§7.2.1"):
        match = next((c for c in clauses if c["id"] == clause_id), None)
        if match is None:
            print(f"{clause_id} : NOT FOUND")
            continue
        has_table = "|" in match["text"]
        print(f"{clause_id} contains table : {has_table}")
        if "--show-tables" in sys.argv:
            print(match["text"])
        print()
