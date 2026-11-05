"""Shared helper: expected service for a block at a given headcount."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def expected_service(model, dept, store, hours_needed, paid_hours, n, p_callout, backfill):
    """P(meet target) averaged over Binomial callouts, plus expected worked staff and utilization.

    worked = scheduled - callouts + backfilled callouts (backfill share from ASSUMPTIONS.md section 3).
    """
    n = int(n)
    if n <= 0:
        return 0.0, 0.0, float("nan")
    if hours_needed <= 0:
        return 1.0, float(n), 0.0
    k = np.arange(n + 1)
    pmf = binom.pmf(k, n, p_callout)
    worked = np.maximum(n - k * (1 - backfill), 0.5)
    util = hours_needed / (worked * paid_hours)
    p = float((pmf * model.p_meet(dept, store, util)).sum())
    ew = float((pmf * worked).sum())
    return p, ew, hours_needed / (ew * paid_hours)


WAIT_CAP = 3.0  # a shift's wait counts at most 3x its target, so one hopeless shift can't absorb all staff


def expected_wait_ratio(model, dept, store, hours_needed, paid_hours, n, p_callout, backfill, target, cap=WAIT_CAP):
    """Expected (service metric / target), averaged over Binomial callouts, each outcome capped at `cap`."""
    n = int(n)
    if n <= 0:
        return cap
    if hours_needed <= 0:
        return 0.0
    k = np.arange(n + 1)
    pmf = binom.pmf(k, n, p_callout)
    worked = np.maximum(n - k * (1 - backfill), 0.5)
    util = hours_needed / (worked * paid_hours)
    ratio = np.minimum(np.asarray(model.expected_metric(dept, util)) / target, cap)
    return float((pmf * ratio).sum())
