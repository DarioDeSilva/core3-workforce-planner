"""Demand forecast for the planning week.

Method (all steps are explainable on one slide):
1. Daily baseline = mean of the same weekday over Normal (non-holiday) days in the chosen weeks,
   per warehouse and department. Holiday-window days are excluded so the MLK weekend does not
   inflate the baseline (the naive benchmark includes it).
2. Holiday lifts from the MLK analog, by department:
   Pre_MLK_Weekend -> Pre_Presidents_Day_Weekend, MLK_Holiday -> Presidents_Day_Holiday,
   Jan 20 rebound -> Feb 17 rebound. Plus an optional Valentine's Day judgment multiplier.
3. Split the day into shifts with historical shift shares by department, day type, and whether the
   day is in a holiday window (Front End demand tilts toward midday on holiday days).
4. Scenario demand = BASE forecast x the scenario's demand multiplier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

HOLIDAY_ROLES_HIST = {"Pre_MLK_Weekend", "MLK_Holiday", "Post_New_Year_Rebound"}
REBOUND_HIST_DATE = "2026-01-20"
VALENTINES = "2026-02-14"


def _daily(h: pd.DataFrame) -> pd.DataFrame:
    cols = ["store_id", "dept_id", "day_date", "day_name", "holiday_window_role", "week_of_year"]
    return h.groupby(cols, as_index=False).shift_demand_units.sum().rename(
        columns={"shift_demand_units": "demand"})


def baseline(h: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
    d = _daily(h)
    normal = d[(d.holiday_window_role == "Normal") & (d.day_date != REBOUND_HIST_DATE)
               & d.week_of_year.isin(weeks)]
    return normal.groupby(["store_id", "dept_id", "day_name"], as_index=False).demand.mean().rename(
        columns={"demand": "baseline"})


def estimate_lifts(h: pd.DataFrame, weeks: list[int] | None = None) -> dict[str, dict[int, float]]:
    """Holiday lifts = actual / same-weekday Normal baseline, averaged over warehouses."""
    weeks = weeks or [2, 3, 4, 5, 6]
    d = _daily(h).merge(baseline(h, weeks), on=["store_id", "dept_id", "day_name"])
    d["lift"] = d.demand / d.baseline
    out = {}
    for name, mask in {
        "pre_holiday_weekend": d.holiday_window_role == "Pre_MLK_Weekend",
        "holiday": d.holiday_window_role == "MLK_Holiday",
        "post_holiday_rebound": d.day_date == REBOUND_HIST_DATE,
    }.items():
        out[name] = {int(k): round(float(v), 4) for k, v in d[mask].groupby("dept_id").lift.mean().items()}
    return out


def shift_shares(h: pd.DataFrame) -> pd.DataFrame:
    o = h[h.is_open == 1].copy()
    o["holiday_window"] = o.holiday_window_role.isin(HOLIDAY_ROLES_HIST)
    s = o.groupby(["dept_id", "day_type", "holiday_window", "shift_code"], as_index=False).shift_share_of_day.mean()
    s["share"] = s.shift_share_of_day / s.groupby(["dept_id", "day_type", "holiday_window"]).shift_share_of_day.transform("sum")
    return s.drop(columns="shift_share_of_day")


def forecast_blocks(params: dict, h: pd.DataFrame | None = None, g: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per scenario x planning-week block with forecast_units (BASE x multiplier)."""
    h = data.history() if h is None else h
    g = data.grid() if g is None else g
    fp = params["forecast"]
    lifts = fp.get("holiday_lifts")
    if lifts in (None, "auto"):
        lifts = estimate_lifts(h, fp["baseline_weeks"])
    vday = fp.get("valentines_extra") or {}
    lift_scale = float(fp.get("holiday_lift_scale", 1.0))

    days = g[g.scenario_code == "BASE"][["store_id", "dept_id", "day_date", "day_name", "day_type",
                                          "holiday_window_role", "is_post_holiday_rebound"]].drop_duplicates()
    days = days.merge(baseline(h, fp["baseline_weeks"]), on=["store_id", "dept_id", "day_name"], how="left")

    def lift(r):
        if r.holiday_window_role == "Pre_Presidents_Day_Weekend":
            m = lifts["pre_holiday_weekend"].get(r.dept_id, 1.0)
        elif r.holiday_window_role == "Presidents_Day_Holiday":
            m = lifts["holiday"].get(r.dept_id, 1.0)
        elif r.is_post_holiday_rebound == 1:
            m = lifts["post_holiday_rebound"].get(r.dept_id, 1.0)
        else:
            m = 1.0
        if lift_scale != 1.0 and m != 1.0:  # sensitivity: weaker/stronger holiday effect than MLK
            m = 1 + (m - 1) * lift_scale
        if r.day_date == VALENTINES:
            m *= vday.get(r.dept_id, 1.0)
        return m

    days["holiday_lift"] = days.apply(lift, axis=1)
    days["daily_forecast"] = days.baseline * days.holiday_lift
    days["holiday_window"] = days.holiday_window_role != "Normal"

    blocks = g[["scenario_id", "scenario_code", "store_id", "dept_id", "day_date", "shift_code",
                "day_type", "is_open", "demand_multiplier"]]
    blocks = blocks.merge(days[["store_id", "dept_id", "day_date", "baseline", "holiday_lift",
                                "daily_forecast", "holiday_window"]], on=["store_id", "dept_id", "day_date"])
    blocks = blocks.merge(shift_shares(h), on=["dept_id", "day_type", "holiday_window", "shift_code"], how="left")
    blocks["share"] = blocks.share.fillna(0.0)
    blocks.loc[blocks.is_open == 0, "share"] = 0.0
    blocks["base_units"] = blocks.daily_forecast * blocks.share
    blocks["forecast_units"] = blocks.base_units * blocks.demand_multiplier
    return blocks


def backtest(h: pd.DataFrame | None = None, test_weeks: tuple[int, ...] = (5, 6)) -> pd.DataFrame:
    """Shift-level MAPE on held-out weeks: our method vs the naive same-weekday mean.

    Train only on weeks before each test week (no peeking)."""
    h = data.history() if h is None else h
    rows = []
    shares = shift_shares(h)
    for wk in test_weeks:
        train = h[h.week_of_year < wk]
        test = h[(h.week_of_year == wk) & (h.is_open == 1)].copy()
        b = baseline(train, list(range(2, wk)))
        naive = _daily(train).groupby(["store_id", "dept_id", "day_name"], as_index=False).demand.mean()
        naive_share = train[train.is_open == 1].groupby(["dept_id", "day_type", "shift_code"], as_index=False).shift_share_of_day.mean()
        t = test.merge(b, on=["store_id", "dept_id", "day_name"]).merge(
            naive.rename(columns={"demand": "naive_daily"}), on=["store_id", "dept_id", "day_name"])
        t["holiday_window"] = t.holiday_window_role.isin(HOLIDAY_ROLES_HIST)
        t = t.merge(shares, on=["dept_id", "day_type", "holiday_window", "shift_code"]).merge(
            naive_share.rename(columns={"shift_share_of_day": "naive_share"}), on=["dept_id", "day_type", "shift_code"])
        t["ours"] = t.baseline * t.share
        t["naive"] = t.naive_daily * t.naive_share
        for m in ["ours", "naive"]:
            t[f"ape_{m}"] = (t[m] - t.shift_demand_units).abs() / t.shift_demand_units
        t["test_week"] = wk
        rows.append(t)
    r = pd.concat(rows)
    return r.groupby(["test_week", "dept_id"]).agg(mape_ours=("ape_ours", "mean"), mape_naive=("ape_naive", "mean")).round(4)
