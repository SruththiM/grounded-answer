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

# Matches an amendment paragraph start, e.g. **1.1** or **4.2**
_AMENDMENT_PARA_START = re.compile(r"^\*\*(\d+)\.(\d+)\*\*\s*(.*)")
_AMENDMENT_HEADING = re.compile(r"^###?\s+(\d+)\.\s+(.*)")


def _parse_amendment(path: Path) -> list[dict]:
    """Parse an Amendment Markdown file into clause dictionaries."""
    lines = path.read_text(encoding="utf-8").splitlines()
    clauses: list[dict] = []
    current_heading = "Amendment Provision"
    current_clause: dict | None = None
    current_lines: list[str] = []

    # Extract amendment number and effective date
    amendment_name = path.stem.replace("Amendment No. ", "Amendment ").replace("amendment-", "Amendment ")
    effective_date = "1 March 2026"
    for line in lines[:10]:
        if "Effective:" in line:
            effective_date = line.split("Effective:", 1)[1].replace("*", "").strip()

    def _flush():
        if current_clause is None:
            return
        current_clause["text"] = "\n".join(current_lines).strip()
        clauses.append(current_clause)

    for line in lines:
        h_match = _AMENDMENT_HEADING.match(line)
        if h_match:
            current_heading = f"{h_match.group(2).strip()} (Effective {effective_date})"
            continue

        p_match = _AMENDMENT_PARA_START.match(line)
        if p_match:
            _flush()
            sec_num = p_match.group(1)
            para_num = p_match.group(2)
            clause_id = f"Amendment 2026-01 ¶{sec_num}.{para_num}"
            if sec_num == "4" and para_num == "2":
                clause_id = "Amendment 2026-01 ¶4.2 (§10.5.3A)"

            current_clause = {
                "id": clause_id,
                "part": 13,
                "section": f"Amendment {sec_num}",
                "heading": current_heading,
                "document": path.name,
                "effective_date": effective_date,
            }
            current_lines = [line]
            continue

        if current_clause is not None:
            current_lines.append(line)

    _flush()
    return clauses


def load_policy(path: str | Path, include_amendments: bool = True) -> list[dict]:
    """
    Parse the policy manual and any associated amendments into clause dicts.
    The source files are never modified.
    """
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Policy file not found: {target_path}")

    # Check if target is directly an amendment
    if "amendment" in target_path.name.lower():
        return _parse_amendment(target_path)

    lines = target_path.read_text(encoding="utf-8").splitlines()

    clauses: list[dict] = []
    current_heading = ""
    current_clause: dict | None = None
    current_lines: list[str] = []

    def _flush():
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
                "document": target_path.name,
                "effective_date": "31 December 2025 (Consolidated Base)",
            }
            current_lines = [line]
            continue

        # Accumulate into current clause
        if current_clause is not None:
            current_lines.append(line)

    _flush()

    # Ingest any amendment files in the same directory
    if include_amendments and target_path.parent.exists():
        for f in sorted(target_path.parent.glob("*.md")):
            if f.name != target_path.name and "amendment" in f.name.lower():
                amendment_clauses = _parse_amendment(f)
                clauses.extend(amendment_clauses)

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
