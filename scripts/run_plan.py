"""Build the staffing plan for all scenarios and write the outputs.

    python scripts/run_plan.py                # optimizer mode (default)
    python scripts/run_plan.py --mode rule    # rule-based mode

Writes:
    outputs/plan_by_block.csv        every block, all scenarios (Mia's Tableau source)
    outputs/scenario_summary.csv     cost and service vs the draft, by scenario
    outputs/Core3Inc_submission.csv  the competition template, filled in
    outputs/cost_service_frontier.csv  (with --frontier) cost vs service curve per scenario
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import data  # noqa: E402
from engine.optimizer import frontier  # noqa: E402
from engine.service import ServiceModel  # noqa: E402
from engine.staffing import build_plan, prepare, summarize  # noqa: E402
from engine.submission import make_submission, validate  # noqa: E402

PLAN_COLS = [
    "scenario_code", "store_id", "dept_id", "department_group", "day_date", "day_name", "shift_code",
    "holiday_window_role", "is_open", "min_staff", "paid_hours", "demand_multiplier",
    "baseline", "holiday_lift", "forecast_units", "hours_needed", "p_callout",
    "available_regular_employees", "cross_trained_flex_employees", "max_overtime_employees",
    "required_employees", "recommended_employees", "capacity_shortfall",
    "rec_regular", "rec_flex", "rec_ot", "rec_capacity", "rec_cost", "rec_feasible",
    "rec_utilization", "rec_p_meet", "rec_expected_metric", "rec_risk",
    "draft_scheduled_employees", "draft_cost", "draft_feasible", "draft_utilization", "draft_p_meet",
    "draft_expected_metric", "draft_risk", "change_vs_draft", "service_target_value", "metric_code", "why", "mode",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rule", "optimizer"], default="optimizer")
    ap.add_argument("--frontier", action="store_true", help="also write outputs/cost_service_frontier.csv (~15 s)")
    args = ap.parse_args()
    params = data.load_params()
    model = ServiceModel.fit()
    prepared = prepare(params)
    plan = build_plan(params, mode=args.mode, model=model, prepared=prepared)
    data.OUTPUTS.mkdir(exist_ok=True)
    plan[PLAN_COLS].sort_values(["scenario_code", "store_id", "dept_id", "day_date", "shift_code"]).to_csv(
        data.OUTPUTS / "plan_by_block.csv", index=False)
    summ = summarize(plan)
    summ.round(3).to_csv(data.OUTPUTS / "scenario_summary.csv")
    sub = make_submission(plan, params["team_id"])
    problems = validate(sub, plan)
    sub.to_csv(data.OUTPUTS / "Core3Inc_submission.csv", index=False)
    if args.frontier:
        import pandas as pd
        backfill = float(params["staffing"].get("backfill_rate", 0.5))
        fr = pd.concat([frontier(prepared[prepared.scenario_code == sc], model, backfill).assign(scenario_code=sc)
                        for sc in ["BASE", "PEAK", "DISRUPT"]])
        fr.round(4).to_csv(data.OUTPUTS / "cost_service_frontier.csv", index=False)
    print(summ.round(2).T.to_string())
    print("\nSubmission check:", "OK" if not problems else problems)


if __name__ == "__main__":
    main()
