"""Load case data and parameters."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import json

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
CONFIG = ROOT / "config" / "parameters.yaml"

FE, MS, RX = 38, 92, 95
DEPT_NAMES = {FE: "Front End", MS: "Merchandising/Stocking", RX: "Pharmacy"}
PARTNER = {FE: MS, MS: FE}  # cross-training pairs; Pharmacy has none
LABOR_STANDARD = {FE: 1.5, MS: 1.2, RX: 0.165}  # overwritten from file in load_standards()
KEY = ["store_id", "dept_id", "day_date", "shift_code"]
SHIFT_ORDER = {"OPEN": 1, "MID": 2, "EVE": 3}


@lru_cache(maxsize=None)
def history() -> pd.DataFrame:
    return pd.read_csv(RAW / "v_ShiftHistory.csv")


@lru_cache(maxsize=None)
def grid() -> pd.DataFrame:
    return pd.read_csv(RAW / "v_PlanningWeekGrid.csv")


@lru_cache(maxsize=None)
def template() -> pd.DataFrame:
    return pd.read_csv(RAW / "TEAM_SUBMISSION_TEMPLATE.csv")


@lru_cache(maxsize=None)
def benchmark() -> pd.DataFrame:
    return pd.read_csv(RAW / "PlanningWeek_BenchmarkForecast.csv")


@lru_cache(maxsize=None)
def standards() -> dict[int, float]:
    df = pd.read_csv(RAW / "comp_LaborStandard.csv")
    return dict(zip(df.dept_id, df.labor_hours_per_workload_unit))


def load_params(path: Path | str = CONFIG) -> dict:
    with open(path) as f:
        p = yaml.safe_load(f)
    # YAML keys like 38 load as ints already; make sure nested dicts use int keys
    for sect, key in [("forecast", "valentines_extra"), ("staffing", "target_utilization"),
                      ("staffing", "service_target_by_dept")]:
        d = p[sect].get(key)
        if isinstance(d, dict):
            p[sect][key] = {int(k): float(v) for k, v in d.items()}
    return p


APP_CACHE = OUTPUTS / "app_cache.pkl"


def canonical_key(params: dict, mode: str) -> str:
    """Stable fingerprint of the settings a plan was built with (used to reuse a precomputed plan)."""
    return json.dumps(params, sort_keys=True, default=str) + "|" + mode
