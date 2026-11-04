"""Official cost and feasibility rule (README "Filling in the staffing plan"; ASSUMPTIONS.md section 5).

For each block: fill with regular available staff first; then cross-trained staff from the partner
department (Front End <-> Merchandising only), limited to partner regular staff that department is NOT
using in the same shift; then overtime. A block above its capacity or below min_staff is infeasible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FE, MS, PARTNER

PAIR_KEY = ["scenario_id", "store_id", "day_date", "shift_code"]


def partner_spare(df: pd.DataFrame, staff_col: str) -> pd.Series:
    """Regular staff the partner department is not using in the same block."""
    spare = (df.available_regular_employees - np.minimum(df[staff_col], df.available_regular_employees)).clip(lower=0)
    tmp = df[PAIR_KEY + ["dept_id"]].assign(spare=spare.values)
    tmp["partner"] = tmp.dept_id.map(PARTNER)
    look = tmp.rename(columns={"dept_id": "pdept", "spare": "partner_spare"})[PAIR_KEY + ["pdept", "partner_spare"]]
    out = tmp.merge(look, left_on=PAIR_KEY + ["partner"], right_on=PAIR_KEY + ["pdept"], how="left")
    return pd.Series(out.partner_spare.fillna(0).values, index=df.index)


def capacity(df: pd.DataFrame, staff_col: str) -> pd.Series:
    usable_flex = np.minimum(df.cross_trained_flex_employees, partner_spare(df, staff_col))
    return df.available_regular_employees + usable_flex + df.max_overtime_employees


def cost_plan(df: pd.DataFrame, staff_col: str, prefix: str) -> pd.DataFrame:
    """Add {prefix}_regular/_flex/_ot/_cost/_capacity/_feasible columns for the staffing in staff_col."""
    out = df.copy()
    n = out[staff_col].astype(int)
    spare = partner_spare(out, staff_col)
    reg = np.minimum(n, out.available_regular_employees)
    flex = np.minimum(n - reg, np.minimum(out.cross_trained_flex_employees, spare))
    ot = np.minimum(n - reg - flex, out.max_overtime_employees)
    cap = out.available_regular_employees + np.minimum(out.cross_trained_flex_employees, spare) + out.max_overtime_employees
    hours = out.paid_hours
    out[f"{prefix}_regular"] = reg.astype(int)
    out[f"{prefix}_flex"] = flex.astype(int)
    out[f"{prefix}_ot"] = ot.astype(int)
    out[f"{prefix}_capacity"] = cap.astype(int)
    # blocks staffed above capacity are costed at their (capped) feasible staff and flagged
    out[f"{prefix}_cost"] = ((reg + flex) * hours * out.loaded_regular_hourly_cost
                             + ot * hours * out.loaded_overtime_hourly_cost).round(2)
    feasible = (out.is_open == 0) & (n == 0) | (out.is_open == 1) & (n >= out.min_staff) & (n <= cap)
    out[f"{prefix}_feasible"] = feasible
    return out
