"""Service-risk model learned from history.

utilization = demand x labor standard / worked labor hours

Two pieces, both fit per department:
* P(meet target | utilization, warehouse): logistic regression with warehouse offsets
  (ASSUMPTIONS.md section 6 says warehouses differ in process). Used for risk and by the optimizer.
* Expected metric value (minutes) vs utilization: median by 0.05 utilization band, interpolated.
  Used only for display ("expected wait ~4 min").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data

STORES = [1, 2, 4]


def history_utilization(h: pd.DataFrame | None = None) -> pd.DataFrame:
    h = data.history() if h is None else h
    std = data.standards()
    o = h[h.is_open == 1].copy()
    o["utilization"] = o.shift_demand_units * o.dept_id.map(std) / o.worked_labor_hours
    return o


def _fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 100) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


@dataclass
class ServiceModel:
    coefs: dict[int, np.ndarray] = field(default_factory=dict)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    targets: dict[int, float] = field(default_factory=dict)

    @staticmethod
    def _design(util, store) -> np.ndarray:
        util = np.atleast_1d(np.asarray(util, dtype=float))
        store = np.broadcast_to(np.atleast_1d(store), util.shape)
        cols = [np.ones_like(util), util] + [(store == s).astype(float) for s in STORES[1:]]
        return np.column_stack(cols)

    @classmethod
    def fit(cls, h: pd.DataFrame | None = None) -> "ServiceModel":
        o = history_utilization(h)
        m = cls()
        for dept, d in o.groupby("dept_id"):
            X = cls._design(d.utilization.values, d.store_id.values)
            m.coefs[int(dept)] = _fit_logistic(X, d.met_target.values.astype(float))
            band = (d.utilization * 20).clip(upper=26).floordiv(1) / 20 + 0.025
            c = d.groupby(band).metric_value.median()
            m.curves[int(dept)] = (c.index.values, np.maximum.accumulate(c.values))
            m.targets[int(dept)] = float(d.target_value.iloc[0])
        return m

    def p_meet(self, dept: int, store, util) -> np.ndarray:
        X = self._design(util, store)
        z = np.clip(X @ self.coefs[int(dept)], -30, 30)
        return 1 / (1 + np.exp(-z))

    def expected_metric(self, dept: int, util) -> np.ndarray:
        x, y = self.curves[int(dept)]
        return np.interp(util, x, y)

    def threshold(self, dept: int, store: int, prob: float) -> float:
        """Highest utilization at which P(meet) >= prob for this warehouse."""
        w = self.coefs[int(dept)]
        X0 = self._design([0.0], store)[0]
        logit = np.log(prob / (1 - prob))
        return float((logit - (X0 @ w)) / w[1])


def callout_rates(h: pd.DataFrame | None = None) -> dict[tuple[int, bool], float]:
    """Historical callout rate by department and weekend flag (weekends run higher)."""
    h = data.history() if h is None else h
    o = h[h.is_open == 1]
    g = o.groupby(["dept_id", "is_weekend"]).agg(c=("callout_employees", "sum"), s=("scheduled_employees", "sum"))
    return {(int(d), bool(w)): float(r.c / r.s) for (d, w), r in g.iterrows()}
