"""Turn forecast demand into recommended headcount per block, then cost and risk-score it.

Two modes:
* "rule"      (Option A) required worked staff at a target utilization, plus a callout buffer.
* "optimizer" (Option B) cost-minimizing allocation (engine/optimizer.py): cheapest plan whose expected
                share of blocks meeting target reaches the portfolio target, with a per-block floor.
Both modes respect minimum staff and each block's capacity, where Front End and Merchandising share
cross-trained staff (one borrowed person is one fewer available to the partner).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import data, forecast
from .costing import capacity, cost_plan
from .optimizer import optimize
from .service import ServiceModel, callout_rates
from .staffing_core import expected_service


def _callout_p(df: pd.DataFrame, rates: dict) -> pd.Series:
    return pd.Series([rates.get((d, bool(w)), 0.05) for d, w in zip(df.dept_id, df.is_weekend)], index=df.index)


def rule_headcount(row, target_util: float, p_callout: float) -> int:
    if row.is_open == 0:
        return 0
    worked = math.ceil(row.hours_needed / (row.paid_hours * target_util) - 1e-9)
    sched = math.ceil(worked / (1 - p_callout) - 1e-9)
    return max(sched, int(row.min_staff))


def prepare(params: dict, scenarios=("BASE", "PEAK", "DISRUPT")) -> pd.DataFrame:
    """Planning grid + forecast + hours of work + callout probability."""
    g = data.grid()
    g = g[g.scenario_code.isin(scenarios)].copy()
    fc = forecast.forecast_blocks(params, g=data.grid())
    df = g.merge(fc[data.KEY + ["scenario_id", "baseline", "holiday_lift", "daily_forecast", "share", "forecast_units"]],
                 on=data.KEY + ["scenario_id"], how="left")
    df["labor_std"] = df.dept_id.map(data.standards())
    df["hours_needed"] = df.forecast_units * df.labor_std
    sp = params["staffing"]
    rates = callout_rates() if sp.get("callout_rates") in (None, "auto") else sp["callout_rates"]
    df["p_callout"] = _callout_p(df, rates)
    return df


def build_plan(params: dict, mode: str = "optimizer", scenarios=("BASE", "PEAK", "DISRUPT"),
               model: ServiceModel | None = None, prepared: pd.DataFrame | None = None) -> pd.DataFrame:
    df = prepare(params, scenarios) if prepared is None else prepared.copy()
    sp = params["staffing"]
    backfill = float(sp.get("backfill_rate", 0.5))
    model = model or ServiceModel.fit()

    if mode == "rule":
        tu = sp["target_utilization"]
        df["required_employees"] = [rule_headcount(r, tu[int(r.dept_id)], r.p_callout) for r in df.itertuples()]
        df["recommended_employees"] = df.required_employees
        for _ in range(2):  # respect joint FE/Merch capacity
            df["recommended_employees"] = np.minimum(df.recommended_employees, capacity(df, "recommended_employees")).astype(int)
    else:
        df["recommended_employees"] = optimize(
            df, model, backfill, float(sp.get("portfolio_service_target", 0.99)),
            float(sp.get("block_service_floor", 0.0)),
            util_ceiling=float(sp.get("utilization_ceiling", 0.90))).astype(int)
        df["required_employees"] = df.recommended_employees
    df["capacity_shortfall"] = (df.required_employees - df.recommended_employees).clip(lower=0)

    df = cost_plan(df, "recommended_employees", "rec")
    df = cost_plan(df, "draft_scheduled_employees", "draft")
    df = score_service(df, model, backfill)
    df["change_vs_draft"] = df.recommended_employees - df.draft_scheduled_employees
    df["why"] = [explain(r, mode, sp) for r in df.itertuples()]
    df["mode"] = mode
    return df


def explain(r, mode: str, sp: dict) -> str:
    """Plain-English reason for one block's recommendation."""
    if r.is_open == 0:
        return "Closed block."
    parts = [f"Forecast {r.forecast_units:.1f} units = {r.hours_needed:.1f} h of work"]
    if r.holiday_lift != 1:
        parts.append(f"holiday lift x{r.holiday_lift:.2f}")
    if r.demand_multiplier != 1:
        parts.append(f"scenario demand x{r.demand_multiplier:.2f}")
    if mode == "rule":
        parts.append(f"staffed to {sp['target_utilization'][int(r.dept_id)]:.2f} utilization + {r.p_callout:.0%} callout buffer")
    else:
        parts.append(f"cheapest headcount in the optimized plan; {r.rec_p_meet:.0%} chance of meeting target after callouts")
    if r.recommended_employees == r.min_staff:
        parts.append("at minimum staff")
    if r.rec_flex > 0:
        parts.append(f"{int(r.rec_flex)} cross-trained from partner dept (cheaper than overtime)")
    if r.rec_ot > 0:
        parts.append(f"{int(r.rec_ot)} on overtime (regular staff exhausted)")
    if r.recommended_employees >= r.rec_capacity and r.rec_p_meet < 0.9:
        parts.append("AT CAPACITY: service risk remains; escalate (borrow from another warehouse, defer tasks)")
    d = int(r.change_vs_draft)
    parts.append(f"{'+' if d > 0 else ''}{d} vs draft ({int(r.draft_scheduled_employees)})")
    return "; ".join(parts) + "."


def score_service(df: pd.DataFrame, model: ServiceModel, backfill: float) -> pd.DataFrame:
    for prefix, col in [("rec", "recommended_employees"), ("draft", "draft_scheduled_employees")]:
        ps, us, ms = [], [], []
        for r in df.itertuples():
            if r.is_open == 0:
                ps.append(np.nan); us.append(np.nan); ms.append(np.nan)
                continue
            # a block staffed above capacity can only field its capacity
            n = int(min(getattr(r, col), getattr(r, f"{prefix}_capacity")))
            p, _, u = expected_service(model, r.dept_id, r.store_id, r.hours_needed, r.paid_hours, n, r.p_callout, backfill)
            ps.append(p); us.append(u); ms.append(float(model.expected_metric(r.dept_id, u)))
        df[f"{prefix}_p_meet"] = ps
        df[f"{prefix}_utilization"] = us
        df[f"{prefix}_expected_metric"] = ms
        df[f"{prefix}_risk"] = pd.cut(df[f"{prefix}_p_meet"], [-0.01, 0.6, 0.9, 1.01], labels=["High", "Medium", "Low"]).astype(str)
        df.loc[df.is_open == 0, f"{prefix}_risk"] = "Closed"
    return df


def summarize(plan: pd.DataFrame, by=("scenario_code",)) -> pd.DataFrame:
    o = plan[plan.is_open == 1].assign(rec_hours=lambda x: x.recommended_employees * x.paid_hours,
                                         draft_hours=lambda x: x.draft_scheduled_employees * x.paid_hours)
    s = o.groupby(list(by)).agg(
        blocks=("recommended_employees", "size"),
        rec_cost=("rec_cost", "sum"), draft_cost=("draft_cost", "sum"),
        rec_hours=("rec_hours", "sum"), draft_hours=("draft_hours", "sum"),
        rec_expected_blocks_meeting=("rec_p_meet", "sum"), draft_expected_blocks_meeting=("draft_p_meet", "sum"),
        rec_high_risk=("rec_risk", lambda x: (x == "High").sum()), draft_high_risk=("draft_risk", lambda x: (x == "High").sum()),
        rec_infeasible=("rec_feasible", lambda x: (~x).sum()), draft_infeasible=("draft_feasible", lambda x: (~x).sum()),
        rec_ot_people=("rec_ot", "sum"), rec_flex_people=("rec_flex", "sum"),
    )
    s["rec_pct_meeting"] = s.rec_expected_blocks_meeting / s.blocks
    s["draft_pct_meeting"] = s.draft_expected_blocks_meeting / s.blocks
    s["cost_change"] = s.rec_cost - s.draft_cost
    s["cost_change_pct"] = s.cost_change / s.draft_cost
    return s
