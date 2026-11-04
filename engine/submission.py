"""Fill and validate TEAM_SUBMISSION_TEMPLATE.csv."""
from __future__ import annotations

import pandas as pd

from . import data


def make_submission(plan: pd.DataFrame, team_id: str) -> pd.DataFrame:
    t = data.template().copy()
    t["team_id"] = team_id
    rec = plan[["scenario_id", "store_id", "dept_id", "day_date", "shift_code", "recommended_employees"]]
    t = t.drop(columns="submitted_employees").merge(rec, on=["scenario_id", "store_id", "dept_id", "day_date", "shift_code"], how="left")
    t = t.rename(columns={"recommended_employees": "submitted_employees"})
    t["submitted_employees"] = t.submitted_employees.fillna(0).astype(int)
    return t[data.template().columns]


def validate(sub: pd.DataFrame, plan: pd.DataFrame) -> list[str]:
    problems = []
    if len(sub) != 567:
        problems.append(f"expected 567 rows, got {len(sub)}")
    if sub.submitted_employees.isna().any():
        problems.append("missing submitted_employees")
    if ((sub.is_open == 0) & (sub.submitted_employees != 0)).any():
        problems.append("closed blocks must be 0")
    bad = plan[(plan.is_open == 1) & ~plan.rec_feasible]
    if len(bad):
        problems.append(f"{len(bad)} infeasible blocks (above capacity or below minimum staff)")
    return problems
