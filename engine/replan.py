"""Mid-week re-plan: react to what is actually happening and re-optimize the rest of the week.

Shifts before `as_of` are already worked and stay locked. For the remaining shifts we apply the manager's
observation (traffic running above/below forecast, extra callouts) and re-run the optimizer. The result is
compared with re-planning the same remaining shifts *without* the change, so the difference is exactly what
the new information requires (the optimizer's goal applies to the remaining shifts in both runs).
Splitting by date keeps Front End / Merchandising pairs together, so cross-training stays valid.
"""
from __future__ import annotations

import pandas as pd

from .compare import diff_plans, summarize_diff
from .service import ServiceModel
from .staffing import build_plan, prepare


def replan(params: dict, model: ServiceModel, scenario: str, as_of: str, store: int, dept: int | None,
           traffic_pct: float = 0.0, extra_callouts: int = 0, mode: str = "optimizer"):
    """Returns (diff, before_plan, after_plan, summary) for the shifts on/after as_of."""
    prepared = prepare(params, (scenario,))
    rest = prepared[prepared.day_date >= as_of].copy()
    before = build_plan(params, mode, model=model, prepared=rest)

    hit = (rest.store_id == store) & (rest.is_open == 1)
    if dept is not None:
        hit &= rest.dept_id == dept
    changed = rest.copy()
    changed.loc[hit, "forecast_units"] = changed.loc[hit, "forecast_units"] * (1 + traffic_pct)
    changed.loc[hit, "hours_needed"] = changed.loc[hit, "forecast_units"] * changed.loc[hit, "labor_std"]
    if extra_callouts:
        changed.loc[hit, "available_regular_employees"] = (changed.loc[hit, "available_regular_employees"]
                                                           - extra_callouts).clip(lower=0)
        note = f"Re-plan: {extra_callouts} extra callouts per shift"
        changed.loc[hit, "availability_note"] = note
    after = build_plan(params, mode, model=model, prepared=changed)

    d = diff_plans(before, after)
    o_b, o_a = before[before.is_open == 1], after[after.is_open == 1]
    summary = summarize_diff(d) | {
        "cost_before": float(before.rec_cost.sum()), "cost_after": float(after.rec_cost.sum()),
        "pass_before": float(o_b.rec_p_meet.mean()), "pass_after": float(o_a.rec_p_meet.mean()),
        "infeasible_after": int((~o_a.rec_feasible).sum()),
    }
    return d, before, after, summary
