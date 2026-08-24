"""
temporal.py — Deterministic temporal resolution engine for policy amendments.

Extracts dates from questions, resolves which policy version applies based on
structured amendment rules and transitional provisions, and identifies when
a date is needed but missing.

This module does NOT call the LLM. All temporal logic is deterministic.

Design:
    Amendments are represented as structured data so that future amendments
    can be added by extending the AMENDMENTS list rather than rewriting code.
"""

import re
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Structured amendment rules
# ---------------------------------------------------------------------------

AMENDMENT_EFFECTIVE_DATE = date(2026, 3, 1)

# Each change within Amendment 2026-01, with its applicability rule.
# applicability_rule:
#   "determination_date"          — applies if determination is on/after effective
#   "change_of_circumstances_date" — applies only if the change occurred on/after effective
AMENDMENT_CHANGES: list[dict[str, Any]] = [
    {
        "amendment_id": "2026-01",
        "paragraph": "1",
        "target_clauses": ["§6.4.1"],
        "description": "Earnings disregard",
        "old_value": "$120 per month",
        "new_value": "$175 per month",
        "applicability_rule": "determination_date",
        "transitional_note": (
            "Applies to any determination made on or after 1 March 2026, "
            "including a determination in respect of a period before that date."
        ),
    },
    {
        "amendment_id": "2026-01",
        "paragraph": "2",
        "target_clauses": ["§4.3.2", "§9.1.4"],
        "description": "Reporting of changes of circumstances",
        "old_value": "10 calendar days (§4.3.2) / 30 calendar days (§9.1.4)",
        "new_value": "14 calendar days",
        "applicability_rule": "change_of_circumstances_date",
        "transitional_note": (
            "Applies only in respect of a change of circumstances occurring "
            "on or after 1 March 2026. Where the change occurred before "
            "1 March 2026, the reporting period is the period that applied "
            "at the date of the change, irrespective of the date of the determination."
        ),
    },
    {
        "amendment_id": "2026-01",
        "paragraph": "3",
        "target_clauses": ["§6.6.1"],
        "description": "Income thresholds",
        "old_value": "HH1=$1,180; HH2=$1,590; HH3=$2,000; HH4=$2,410; HH5=$2,820; +$410",
        "new_value": "HH1=$1,225; HH2=$1,650; HH3=$2,075; HH4=$2,500; HH5=$2,925; +$425",
        "applicability_rule": "determination_date",
        "transitional_note": (
            "Applies to any determination made on or after 1 March 2026, "
            "including a determination in respect of a period before that date."
        ),
    },
    {
        "amendment_id": "2026-01",
        "paragraph": "4",
        "target_clauses": ["§10.5.2", "§10.5.3A"],
        "description": "Sanctions",
        "old_value": "20 per cent (no §10.5.3A existed)",
        "new_value": "15 per cent; new §10.5.3A exemption",
        "applicability_rule": "determination_date",
        "transitional_note": (
            "Applies to any determination made on or after 1 March 2026."
        ),
    },
]

# Clauses affected by any amendment — used to detect when dates matter.
AMENDED_CLAUSES: set[str] = set()
for _change in AMENDMENT_CHANGES:
    for _clause in _change["target_clauses"]:
        AMENDED_CLAUSES.add(_clause)

# Keywords strongly associated with each amended provision.
# Used to detect whether a question touches an amended area.
_AMENDED_TOPIC_PATTERNS: list[dict[str, Any]] = [
    {
        "pattern": re.compile(
            r"\b(earnings?\s+disregard|disregard.*earn|"
            r"employment\s+earnings|earnings?\s+from\s+employment)\b",
            re.IGNORECASE,
        ),
        "target_clauses": ["§6.4.1"],
        "description": "earnings disregard",
        "controlling_date_type": "determination_date",
    },
    {
        "pattern": re.compile(
            r"\b(report(ing)?\s+(change|period|deadline|days)|"
            r"change\s+of\s+circumstances.*days|"
            r"days.*report.*change|"
            r"how\s+(many|long).*days.*report|"
            r"how\s+(many|long).*report|"
            r"report.*change\s+of\s+circumstances)\b",
            re.IGNORECASE,
        ),
        "target_clauses": ["§4.3.2", "§9.1.4"],
        "description": "reporting of changes of circumstances",
        "controlling_date_type": "change_of_circumstances_date",
    },
    {
        "pattern": re.compile(
            r"\b(income\s+threshold|monthly\s+threshold|"
            r"threshold.*household|household.*threshold)\b",
            re.IGNORECASE,
        ),
        "target_clauses": ["§6.6.1"],
        "description": "income thresholds",
        "controlling_date_type": "determination_date",
    },
    {
        "pattern": re.compile(
            r"\b(sanction|sanctions?|penalty|"
            r"per\s*cent.*reduction|reduction.*per\s*cent|"
            r"failure\s+to\s+report.*sanction|"
            r"sanction.*failure\s+to\s+report)\b",
            re.IGNORECASE,
        ),
        "target_clauses": ["§10.5.2", "§10.5.3A"],
        "description": "sanctions",
        "controlling_date_type": "determination_date",
    },
]


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

# Patterns for explicit dates like "15 April 2026", "1 March 2026"
_DATE_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)

# Patterns for month-year references like "February 2026", "April 2026"
_DATE_MONTH_YEAR = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)

# Patterns for date type markers
_DETERMINATION_MARKER = re.compile(
    r"\b(determination|determined|claim\s+dated)\b",
    re.IGNORECASE,
)
_CHANGE_MARKER = re.compile(
    r"\b(change\s+of\s+circumstances|circumstances?\s+changed|change\s+occurred|change\s+occurring)\b",
    re.IGNORECASE,
)
_PERIOD_MARKER = re.compile(
    r"\b(period\s+(?:from|spanning|covering|between)|"
    r"claim\s+(?:period|spanning|covering|from))\b",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_month_year(month_str: str, year_str: str) -> date:
    """Parse a month name and year into a date (1st of that month)."""
    month = _MONTH_MAP[month_str.lower()]
    year = int(year_str)
    return date(year, month, 1)


def _parse_day_month_year(day_str: str, month_str: str, year_str: str) -> date:
    """Parse a day, month name and year into a date."""
    day = int(day_str)
    month = _MONTH_MAP[month_str.lower()]
    year = int(year_str)
    return date(year, month, day)


class ExtractedDates:
    """Container for dates extracted from a question."""

    def __init__(self) -> None:
        self.determination_date: date | None = None
        self.change_of_circumstances_date: date | None = None
        self.period_start: date | None = None
        self.period_end: date | None = None
        self.generic_dates: list[date] = []
        self.is_spanning: bool = False
        self.raw_date_strings: list[str] = []

    def has_any_date(self) -> bool:
        return bool(
            self.determination_date
            or self.change_of_circumstances_date
            or self.period_start
            or self.generic_dates
        )

    def get_controlling_date(self, date_type: str) -> date | None:
        """Get the date that controls applicability for a given rule type."""
        if date_type == "determination_date":
            return self.determination_date or (
                self.generic_dates[0] if self.generic_dates else None
            )
        elif date_type == "change_of_circumstances_date":
            return self.change_of_circumstances_date
        return None

    def __repr__(self) -> str:
        parts = []
        if self.determination_date:
            parts.append(f"determination={self.determination_date}")
        if self.change_of_circumstances_date:
            parts.append(f"change={self.change_of_circumstances_date}")
        if self.period_start:
            parts.append(f"period_start={self.period_start}")
        if self.period_end:
            parts.append(f"period_end={self.period_end}")
        if self.generic_dates:
            parts.append(f"generic={self.generic_dates}")
        if self.is_spanning:
            parts.append("SPANNING")
        return f"ExtractedDates({', '.join(parts)})"


def extract_dates(question: str) -> ExtractedDates:
    """
    Extract temporal information from a question.

    Identifies:
    - Determination dates (e.g., "determination made on 15 April 2026")
    - Change-of-circumstances dates (e.g., "circumstances changed on 20 February 2026")
    - Claim period dates (e.g., "claim dated February 2026")
    - Spanning periods (e.g., "period from February to April 2026")
    - Generic date references (e.g., "in April 2026")
    """
    result = ExtractedDates()

    if not question:
        return result

    # Find all explicit day-month-year dates
    explicit_dates: list[tuple[str, date]] = []
    for m in _DATE_DAY_MONTH_YEAR.finditer(question):
        d = _parse_day_month_year(m.group(1), m.group(2), m.group(3))
        explicit_dates.append((m.group(0), d))
        result.raw_date_strings.append(m.group(0))

    # Find month-year dates (only if not part of a day-month-year already captured)
    month_year_dates: list[tuple[str, date]] = []
    for m in _DATE_MONTH_YEAR.finditer(question):
        full_match = m.group(0)
        # Check if this is part of an explicit date already captured
        is_part_of_explicit = False
        for raw_str, _ in explicit_dates:
            if full_match in raw_str:
                is_part_of_explicit = True
                break
        if not is_part_of_explicit:
            d = _parse_month_year(m.group(1), m.group(2))
            month_year_dates.append((full_match, d))
            result.raw_date_strings.append(full_match)

    all_dates = explicit_dates + month_year_dates

    if not all_dates:
        return result

    # Classify dates by type based on context markers
    has_determination = bool(_DETERMINATION_MARKER.search(question))
    has_change = bool(_CHANGE_MARKER.search(question))
    has_period = bool(_PERIOD_MARKER.search(question))

    if has_determination and has_change and len(all_dates) >= 2:
        # Question mentions both determination and change with multiple dates.
        # Try to assign based on proximity to markers.
        det_pos = _DETERMINATION_MARKER.search(question).start()
        change_pos = _CHANGE_MARKER.search(question).start()

        # Find which date is closer to each marker
        for raw_str, d in all_dates:
            date_pos = question.find(raw_str)
            dist_to_det = abs(date_pos - det_pos)
            dist_to_change = abs(date_pos - change_pos)
            if dist_to_change < dist_to_det:
                if result.change_of_circumstances_date is None:
                    result.change_of_circumstances_date = d
                else:
                    result.generic_dates.append(d)
            else:
                if result.determination_date is None:
                    result.determination_date = d
                else:
                    result.generic_dates.append(d)
    elif has_determination:
        # Only determination context
        for i, (raw_str, d) in enumerate(all_dates):
            if i == 0:
                result.determination_date = d
            else:
                result.generic_dates.append(d)
    elif has_change:
        # Only change-of-circumstances context
        for i, (raw_str, d) in enumerate(all_dates):
            if i == 0:
                result.change_of_circumstances_date = d
            elif has_determination:
                result.determination_date = d
            else:
                result.generic_dates.append(d)
    else:
        # No explicit context markers — treat as generic / claim date
        # "For a claim dated February 2026" => determination date
        if re.search(r"\bclaim\s+dated\b", question, re.IGNORECASE):
            result.determination_date = all_dates[0][1]
            for _, d in all_dates[1:]:
                result.generic_dates.append(d)
        else:
            for _, d in all_dates:
                result.generic_dates.append(d)

    # Detect spanning periods
    if has_period or re.search(r"\bspanning\b", question, re.IGNORECASE):
        result.is_spanning = True
        if len(all_dates) >= 2:
            sorted_dates = sorted([d for _, d in all_dates])
            result.period_start = sorted_dates[0]
            result.period_end = sorted_dates[-1]
            # A period spans the effective date if start < effective <= end
            if result.period_start < AMENDMENT_EFFECTIVE_DATE <= result.period_end:
                result.is_spanning = True
    elif len(all_dates) >= 2:
        sorted_dates = sorted([d for _, d in all_dates])
        if sorted_dates[0] < AMENDMENT_EFFECTIVE_DATE <= sorted_dates[-1]:
            # Dates straddle the amendment effective date
            result.is_spanning = True
            result.period_start = sorted_dates[0]
            result.period_end = sorted_dates[-1]

    return result


# ---------------------------------------------------------------------------
# Amendment applicability resolution
# ---------------------------------------------------------------------------

class TemporalResolution:
    """Result of resolving which policy version applies."""

    def __init__(self) -> None:
        self.applies_amendment: bool | None = None  # None = unknown/needs date
        self.controlling_date: date | None = None
        self.controlling_date_type: str = ""
        self.amendment_changes: list[dict[str, Any]] = []
        self.is_spanning: bool = False
        self.needs_date_refusal: bool = False
        self.refusal_date_type: str = ""
        self.touches_amended_provision: bool = False
        self.explanation: str = ""

    def __repr__(self) -> str:
        if self.needs_date_refusal:
            return f"TemporalResolution(NEEDS_DATE: {self.refusal_date_type})"
        if self.is_spanning:
            return "TemporalResolution(SPANNING)"
        status = "AMENDED" if self.applies_amendment else "BASE"
        return f"TemporalResolution({status}, date={self.controlling_date})"


def _question_touches_amended_topic(question: str) -> list[dict[str, Any]]:
    """
    Check if a question touches any topic that was affected by an amendment.
    Returns matching topic patterns.
    """
    matches = []
    for topic in _AMENDED_TOPIC_PATTERNS:
        if topic["pattern"].search(question):
            matches.append(topic)
    return matches


def resolve_temporal(
    question: str,
    extracted_dates: ExtractedDates,
    evidence: list[dict[str, Any]],
) -> TemporalResolution:
    """
    Determine which policy version applies to a question.

    This is the core deterministic temporal resolution logic.

    Returns a TemporalResolution describing:
    - Whether the amendment applies
    - Which date controlled the decision
    - Whether the question spans the amendment boundary
    - Whether a date is needed but missing (triggering refusal)
    """
    resolution = TemporalResolution()

    # 1. Does the question touch an amended provision?
    amended_topics = _question_touches_amended_topic(question)

    # Also check if retrieved evidence includes amended clauses
    evidence_clause_ids = {c.get("id", "") for c in evidence}
    evidence_touches_amendment = bool(
        evidence_clause_ids & AMENDED_CLAUSES
        or any("Amendment" in c.get("id", "") for c in evidence)
    )

    if not amended_topics and not evidence_touches_amendment:
        # Question does not touch any amended provision — base manual always applies
        resolution.touches_amended_provision = False
        resolution.applies_amendment = False
        resolution.explanation = (
            "This question does not concern a provision affected by "
            "Amendment No. 2026-01. The consolidated manual applies."
        )
        return resolution

    resolution.touches_amended_provision = True

    # 2. Handle spanning periods
    if extracted_dates.is_spanning:
        resolution.is_spanning = True
        resolution.explanation = (
            "The claim period spans 1 March 2026. Under transitional "
            "provision ¶5.3, the applicable figures are those in force "
            "on each day of the period, and the award is apportioned "
            "accordingly under §7.4.3."
        )
        return resolution

    # 3. Determine the controlling date type for the relevant provision
    if amended_topics:
        primary_topic = amended_topics[0]
        controlling_date_type = primary_topic["controlling_date_type"]
    else:
        # Default to determination_date for most provisions
        controlling_date_type = "determination_date"

    resolution.controlling_date_type = controlling_date_type

    # 4. Get the controlling date
    controlling_date = extracted_dates.get_controlling_date(controlling_date_type)

    # If we need the change_of_circumstances_date but only have a
    # determination_date, we CANNOT use the determination date for
    # reporting-period questions (this is the critical transitional rule).
    if controlling_date_type == "change_of_circumstances_date" and controlling_date is None:
        # Check if the question provides a determination date but NOT a change date
        if extracted_dates.determination_date is not None:
            # We have a determination date but not a change date.
            # For reporting questions, we CANNOT use determination date.
            resolution.needs_date_refusal = True
            resolution.refusal_date_type = "change of circumstances"
            resolution.explanation = (
                "The reporting period depends on the date the change of "
                "circumstances occurred, not the determination date. "
                "Amendment No. 2026-01 ¶5.2 requires the change-of-circumstances "
                "date to determine which reporting rule applies."
            )
            return resolution
        elif not extracted_dates.has_any_date():
            # No dates at all
            resolution.needs_date_refusal = True
            resolution.refusal_date_type = "change of circumstances"
            resolution.explanation = (
                "The applicable reporting period depends on when the change "
                "of circumstances occurred. Amendment No. 2026-01 changed "
                "the reporting deadline from the pre-existing rules to "
                "14 calendar days, but this applies only to changes "
                "occurring on or after 1 March 2026."
            )
            return resolution

    if controlling_date_type == "determination_date" and controlling_date is None:
        if not extracted_dates.has_any_date():
            # No dates at all and the question touches an amended provision
            resolution.needs_date_refusal = True
            resolution.refusal_date_type = "determination"
            resolution.explanation = (
                "The applicable policy depends on the date of determination. "
                "Amendment No. 2026-01 changed this provision effective "
                "1 March 2026."
            )
            return resolution

    if controlling_date is None:
        # Last fallback: try generic dates
        if extracted_dates.generic_dates:
            controlling_date = extracted_dates.generic_dates[0]
        else:
            resolution.needs_date_refusal = True
            resolution.refusal_date_type = controlling_date_type.replace("_", " ")
            resolution.explanation = (
                f"The applicable policy depends on the {controlling_date_type.replace('_', ' ')}."
            )
            return resolution

    resolution.controlling_date = controlling_date

    # 5. Determine if the amendment applies
    if controlling_date >= AMENDMENT_EFFECTIVE_DATE:
        resolution.applies_amendment = True
        # Find which specific changes apply
        for change in AMENDMENT_CHANGES:
            if any(tc in [c for t in amended_topics for c in t["target_clauses"]]
                   for tc in change["target_clauses"]):
                resolution.amendment_changes.append(change)

        if controlling_date_type == "determination_date":
            resolution.explanation = (
                f"The determination date ({controlling_date.strftime('%d %B %Y')}) "
                f"is on or after 1 March 2026. Amendment No. 2026-01 applies "
                f"to this determination (transitional provision ¶5.1)."
            )
        else:
            resolution.explanation = (
                f"The change of circumstances occurred on "
                f"{controlling_date.strftime('%d %B %Y')}, which is on or after "
                f"1 March 2026. The amended reporting deadline of 14 calendar "
                f"days applies (transitional provision ¶5.2)."
            )
    else:
        resolution.applies_amendment = False
        if controlling_date_type == "determination_date":
            resolution.explanation = (
                f"The determination date ({controlling_date.strftime('%d %B %Y')}) "
                f"is before 1 March 2026. The consolidated manual (pre-amendment) "
                f"provisions apply."
            )
        else:
            resolution.explanation = (
                f"The change of circumstances occurred on "
                f"{controlling_date.strftime('%d %B %Y')}, which is before "
                f"1 March 2026. Under transitional provision ¶5.2, the "
                f"reporting period that applied at the date of the change "
                f"governs, irrespective of the date of the determination."
            )

    return resolution


# ---------------------------------------------------------------------------
# Temporal context block for the LLM prompt
# ---------------------------------------------------------------------------

def build_temporal_context(
    resolution: TemporalResolution,
    extracted_dates: ExtractedDates,
) -> str:
    """
    Build a structured temporal context block to inject into the LLM prompt.

    This tells the LLM exactly which policy version to use and why,
    so it does not need to reason about temporal applicability itself.
    """
    if not resolution.touches_amended_provision:
        return (
            "TEMPORAL CONTEXT: This question does not concern a provision "
            "affected by Amendment No. 2026-01. Apply the consolidated "
            "policy manual provisions."
        )

    if resolution.needs_date_refusal:
        return (
            "TEMPORAL CONTEXT: REFUSE THIS QUESTION.\n"
            f"Reason: {resolution.explanation}\n"
            "You must refuse to answer and explain that the applicable policy "
            "depends on the relevant date. Ask the user to provide the "
            f"{resolution.refusal_date_type} date.\n"
            "Format your refusal as:\n"
            "  Unable to determine from the policy manual.\n"
            "  [Explanation of why the date matters]\n"
            "  Next step: Please provide the relevant "
            f"{resolution.refusal_date_type} date so the applicable "
            "policy can be identified."
        )

    if resolution.is_spanning:
        return (
            "TEMPORAL CONTEXT: SPANNING PERIOD.\n"
            f"{resolution.explanation}\n"
            "The question relates to a period spanning 1 March 2026. "
            "You must explain that figures differ on each side of 1 March 2026 "
            "and the award must be apportioned under §7.4.3. Cite both the "
            "pre-amendment values and the amended values, and cite "
            "Amendment No. 2026-01 transitional provision ¶5.3."
        )

    if resolution.applies_amendment:
        return (
            "TEMPORAL CONTEXT: AMENDMENT APPLIES.\n"
            f"{resolution.explanation}\n"
            "Use the AMENDED provisions from Amendment No. 2026-01. "
            "Cite both the original clause and the amendment that modifies it. "
            "Show the effective (amended) value, not the pre-amendment value."
        )
    else:
        return (
            "TEMPORAL CONTEXT: PRE-AMENDMENT RULES APPLY.\n"
            f"{resolution.explanation}\n"
            "Use the CONSOLIDATED MANUAL (base) provisions. "
            "Do NOT apply Amendment No. 2026-01 to this question."
        )


def build_refusal_answer(resolution: TemporalResolution) -> str:
    """
    Build a structured refusal answer when a date is needed but missing.
    """
    return (
        "Unable to determine from the policy manual.\n\n"
        "The applicable policy depends on the relevant date, but the question "
        "does not provide enough date information to determine which provision "
        "applies. Amendment No. 2026-01 changed this provision effective "
        "1 March 2026, and different rules may apply depending on the date.\n\n"
        f"Next step: Please provide the relevant "
        f"{resolution.refusal_date_type} date so the applicable policy "
        f"can be identified."
    )
