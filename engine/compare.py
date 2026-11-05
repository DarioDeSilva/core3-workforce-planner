"""Compare two staffing plans and turn the differences into manager actions.

Used for "What changed?" between scenarios (e.g. BASE -> PEAK) and for the mid-week re-plan.
"""
from __future__ import annotations

import pandas as pd

from .data import DEPT_NAMES, FE, MS

KEY = ["store_id", "dept_id", "day_date", "shift_code"]
SHORT_DEPT = {38: "Front End", 92: "Merch", 95: "Pharmacy"}
PARTNER_NAME = {FE: "Merch", MS: "Front End"}


def _direct_reasons(a, b) -> list[str]:
    parts = []
    if abs(b.demand_multiplier - a.demand_multiplier) > 1e-9:
        why = b.adjustment_reason if isinstance(b.adjustment_reason, str) else "scenario"
        parts.append(f"demand x{b.demand_multiplier:.2f} ({why})")
    if abs(b.forecast_units - a.forecast_units) > 0.05 and abs(b.demand_multiplier - a.demand_multiplier) < 1e-9:
        parts.append(f"forecast {a.forecast_units:.0f} -> {b.forecast_units:.0f} units")
    if int(b.available_regular_employees) != int(a.available_regular_employees) or \
            int(b.max_overtime_employees) != int(a.max_overtime_employees):
        note = b.availability_note if isinstance(b.availability_note, str) else "availability changed"
        parts.append(f"{note} (regular {int(a.available_regular_employees)} -> {int(b.available_regular_employees)}, "
                     f"overtime {int(a.max_overtime_employees)} -> {int(b.max_overtime_employees)})")
    return parts


def action_text(r) -> str:
    verb = "add" if r.change > 0 else ("cut" if r.change < 0 else "keep")
    bits = []
    if r.d_ot:
        bits.append(f"{r.d_ot:+d} overtime")
    if r.d_flex:
        bits.append(f"{r.d_flex:+d} borrowed from {PARTNER_NAME.get(int(r.dept_id), 'partner')}")
    if r.d_regular:
        bits.append(f"{r.d_regular:+d} regular")
    split = f" ({', '.join(bits)})" if bits else ""
    sign = "+" if r.d_cost >= 0 else "-"
    return (f"W{int(r.store_id)} {SHORT_DEPT[int(r.dept_id)]} {r.day_name[:3]} {r.day_date[5:]} {r.shift_code}: "
            f"{verb} {abs(int(r.change)) if r.change else int(r.after)}{split}, {sign}${abs(r.d_cost):,.0f}")


def diff_plans(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Shifts where plan b's headcount differs from plan a's (b is the new situation)."""
    cols = KEY + ["day_name", "paid_hours", "recommended_employees", "rec_regular", "rec_flex", "rec_ot", "rec_cost",
                  "rec_p_meet", "demand_multiplier", "adjustment_reason", "forecast_units",
                  "available_regular_employees", "max_overtime_employees", "availability_note", "rec_capacity"]
    m = a[cols].merge(b[cols], on=KEY, suffixes=("_a", "_b"))
    m = m[(m.recommended_employees_a != m.recommended_employees_b) | ((m.rec_cost_a - m.rec_cost_b).abs() > 0.01)].copy()
    out = pd.DataFrame({
        "store_id": m.store_id, "dept_id": m.dept_id, "department": m.dept_id.map(DEPT_NAMES),
        "day_date": m.day_date, "day_name": m.day_name_b, "shift_code": m.shift_code,
        "before": m.recommended_employees_a.astype(int), "after": m.recommended_employees_b.astype(int),
        "change": (m.recommended_employees_b - m.recommended_employees_a).astype(int),
        "d_regular": (m.rec_regular_b - m.rec_regular_a).astype(int),
        "d_flex": (m.rec_flex_b - m.rec_flex_a).astype(int),
        "d_ot": (m.rec_ot_b - m.rec_ot_a).astype(int),
        "d_hours": (m.recommended_employees_b - m.recommended_employees_a) * m.paid_hours_b,
        "d_cost": (m.rec_cost_b - m.rec_cost_a).round(2),
        "p_before": m.rec_p_meet_a, "p_after": m.rec_p_meet_b,
        "at_capacity": m.recommended_employees_b >= m.rec_capacity_b,
    })
    direct = [_direct_reasons(ra, rb) for (_, ra), (_, rb) in
              zip(m.filter(like="_a").rename(columns=lambda c: c[:-2]).iterrows(),
                  m.filter(like="_b").rename(columns=lambda c: c[:-2]).iterrows())]
    out["direct"] = [bool(x) for x in direct]
    shift_key = list(zip(out.store_id, out.day_date, out.shift_code))
    direct_shifts = {k for k, d in zip(shift_key, out.direct) if d}
    reasons = []
    for k, d, dept, chg in zip(shift_key, direct, out.dept_id, out.change):
        if d:
            reasons.append("; ".join(d))
        elif int(dept) in PARTNER_NAME and k in direct_shifts:
            partner = PARTNER_NAME[int(dept)]
            reasons.append(f"knock-on: lends cross-trained staff to {partner}, which is hit by the scenario" if chg < 0
                           else f"knock-on: {partner} needs fewer of the cross-trained staff it borrowed")
        else:
            reasons.append("rebalanced: the week-wide service goal moves staff to where each dollar buys the most service")
    out["reason"] = reasons
    out["action"] = [action_text(r) for r in out.itertuples()]
    return out.sort_values(["day_date", "store_id", "dept_id", "shift_code"]).reset_index(drop=True)


def diff_scenarios(plan: pd.DataFrame, from_code: str, to_code: str) -> pd.DataFrame:
    return diff_plans(plan[plan.scenario_code == from_code], plan[plan.scenario_code == to_code])


def summarize_diff(d: pd.DataFrame) -> dict:
    return {"shifts": int((d.change != 0).sum()), "direct": int(d.direct.sum()), "staff_change": int(d.change.sum()), "hours": float(d.d_hours.sum()),
            "cost": float(d.d_cost.sum()), "added": int(d.change.clip(lower=0).sum()),
            "removed": int(-d.change.clip(upper=0).sum())}
