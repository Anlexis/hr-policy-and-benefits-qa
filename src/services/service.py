"""AgentCore Platform v1.0"""

# Domain service — the entitlement arithmetic behind the answer.
#
# Pure functions over already-validated caller records: no I/O, no credentials,
# no routing. The pre_process node owns the validation contract; everything
# here may assume each record carries a label and two finite day counts within
# the declared bounds, and every function is deterministic, so the same
# submission always yields the same position.

from __future__ import annotations

from typing import Any, Dict, List

# Outcome grades, ordered from most to least severe. The overall grade of a
# submission is the most severe grade any single record carries.
GRADE_SHORTFALL = "shortfall"
GRADE_ATTENTION = "attention"
GRADE_COMPLIANT = "compliant"

_GRADE_ORDER = (GRADE_SHORTFALL, GRADE_ATTENTION, GRADE_COMPLIANT)


def days_display(value: float) -> str:
    """Render a day count the way the answer document quotes it.

    Whole days print without a decimal point; half days keep one decimal
    place. Day counts are exact figures taken from the caller's own record —
    this template computes no monetary value and applies no rounding grid.
    """
    rounded = round(float(value), 1)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.1f}"


def grade_record(
    days_entitled: float,
    days_taken: float,
    statutory_trigger_days: float,
    statutory_minimum_days: float,
    carryover_cap_days: float,
) -> str:
    """Grade one entitlement record.

    ``shortfall`` — the employment regulations require a minimum number of days
    to be taken within the year once an entitlement reaches the trigger size,
    and fewer than that have been taken.
    ``attention`` — the untaken balance is above the carry-over cap, so part of
    it expires at year end.
    ``compliant`` — neither applies.

    The trigger is a threshold on the entitlement itself, so a small
    entitlement is never graded a shortfall for the same untaken balance that
    would be one on a full entitlement.
    """
    remaining = days_entitled - days_taken
    if days_entitled >= statutory_trigger_days and days_taken < statutory_minimum_days:
        return GRADE_SHORTFALL
    if remaining > carryover_cap_days:
        return GRADE_ATTENTION
    return GRADE_COMPLIANT


def summarise_entitlements(
    records: List[Dict[str, Any]],
    statutory_trigger_days: float,
    statutory_minimum_days: float,
    carryover_cap_days: float,
) -> Dict[str, Any]:
    """Compute the employee's entitlement position from validated records.

    Returns ``{"records": [...], "totals": {...}, "overall": <grade>}``. Each
    returned record carries the caller's own label and day counts plus the
    derived remaining balance and grade; the totals are the sums across every
    record. An empty submission returns zero totals and the ``compliant``
    grade, which is what the answer then reports — the policy answer itself is
    unaffected either way.
    """
    graded: List[Dict[str, Any]] = []
    total_entitled = 0.0
    total_taken = 0.0

    for record in records:
        entitled = float(record["days_entitled"])
        taken = float(record["days_taken"])
        remaining = round(entitled - taken, 1)
        graded.append(
            {
                "category": record["category"],
                "policy_reference": record.get("policy_reference", ""),
                "days_entitled": entitled,
                "days_taken": taken,
                "days_remaining": remaining,
                "grade": grade_record(
                    entitled,
                    taken,
                    statutory_trigger_days,
                    statutory_minimum_days,
                    carryover_cap_days,
                ),
            }
        )
        total_entitled += entitled
        total_taken += taken

    overall = GRADE_COMPLIANT
    for grade in _GRADE_ORDER:
        if any(entry["grade"] == grade for entry in graded):
            overall = grade
            break

    return {
        "records": graded,
        "totals": {
            "record_count": len(graded),
            "days_entitled": round(total_entitled, 1),
            "days_taken": round(total_taken, 1),
            "days_remaining": round(total_entitled - total_taken, 1),
        },
        "overall": overall,
    }
