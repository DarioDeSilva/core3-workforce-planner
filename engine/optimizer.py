"""Option B: cost-minimizing staffing optimizer.

Problem, per scenario:
    minimize   total labor cost (official rule: regular -> cross-trained -> overtime)
    subject to expected share of blocks meeting service target >= portfolio_target
               every open block has P(meet) >= block_floor (when capacity allows)
               min_staff <= staff <= capacity, with Front End / Merchandising sharing cross-trained staff

Options (all off by default): per-department goals, a weekly budget instead of a goal, and a wait-time
objective instead of pass rate. They change only the Phase 2 stop rule / eligibility / value table.

Never abandon a shift: once Phase 0 has staffed a shift to a safe level, later moves can add people anywhere but
cannot pull a borrowed person back out of it (so a surge shift never loses staff to help another shift).

Method: (0) staff every block at least to a utilization ceiling (or capacity) so no block is abandoned;
then greedy marginal analysis. P(meet) per block is precomputed for every headcount (averaging over
random callouts). Starting from minimum staff, we (1) raise every block to the floor, then (2) keep adding
the single person with the best "service gained per dollar" until the portfolio target is reached.
Because adding staff has diminishing returns, this greedy rule is close to optimal and it is easy to
explain: every person we add is the cheapest available improvement in service.

Cost is evaluated per FE/Merch pair with the official rule, so borrowing a cross-trained person correctly
shows up as cheaper than overtime, and growing the partner department correctly removes that option.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FE, MS, RX
from .service import ServiceModel
from .staffing_core import WAIT_CAP, expected_service, expected_wait_ratio


def _block_cost(n, reg, flex_spare, flex, ot, hours, r_reg, r_ot):
    a = min(n, reg)
    f = min(n - a, min(flex, flex_spare))
    o = n - a - f
    return (a + f) * hours * r_reg + o * hours * r_ot, o <= ot


class _Unit:
    """One cost unit: a Pharmacy block, or a Front End + Merchandising pair in the same shift."""

    def __init__(self, rows: list):
        self.rows = rows  # list of dict-like rows (1 or 2)

    def cost_ok(self, ns: list[int]):
        if len(self.rows) == 1:
            r = self.rows[0]
            c, ok = _block_cost(ns[0], r["reg"], 0, 0, r["ot"], r["hours"], r["r_reg"], r["r_ot"])
            return c, ok and ns[0] >= r["min"] or ns[0] == 0 and r["closed"]
        total, ok = 0.0, True
        for i in (0, 1):
            r, p = self.rows[i], self.rows[1 - i]
            spare = max(p["reg"] - min(ns[1 - i], p["reg"]), 0)
            c, o = _block_cost(ns[i], r["reg"], spare, r["flex"], r["ot"], r["hours"], r["r_reg"], r["r_ot"])
            total += c
            ok &= o and (ns[i] >= r["min"] or (r["closed"] and ns[i] == 0))
        return total, ok


def optimize(df: pd.DataFrame, model: ServiceModel, backfill: float, portfolio_target: float,
             block_floor: float, max_extra: int = 60, min_gain: float = 0.005, lookahead: int = 0,
             util_ceiling: float = 0.90, dept_targets: dict | None = None, budget=None,
             objective: str = "pass_rate") -> pd.Series:
    """Return recommended headcount for every row of df (one scenario or several).

    dept_targets: {dept_id: goal} -> each department must reach its own pass-rate goal (else one portfolio goal).
    budget: weekly $ cap per scenario (number, or {scenario_code: number}); the optimizer then spends up to the
            budget on the best service per dollar and ignores the service goal.
    objective: "pass_rate" (chance of hitting target) or "wait_time" (expected minutes vs target, capped at 3x).
    The returned Series carries .attrs["budget_shortfall"] = {scenario_code: minimum safe cost} for any scenario
    whose never-abandon-a-shift staffing already costs more than the budget.
    """
    out = pd.Series(0, index=df.index, dtype=int)
    shortfall = {}
    for _, sdf in df.groupby("scenario_id"):
        code = str(sdf.scenario_code.iloc[0]) if "scenario_code" in sdf else ""
        b = budget.get(code) if isinstance(budget, dict) else budget
        res, short = _optimize_scenario(sdf, model, backfill, portfolio_target, block_floor, max_extra, min_gain,
                                        lookahead, util_ceiling, dept_targets, b, objective)
        out.loc[sdf.index] = res
        if short is not None:
            shortfall[code] = short
    out.attrs["budget_shortfall"] = shortfall
    return out


def _optimize_scenario(sdf, model, backfill, portfolio_target, block_floor, max_extra, min_gain, lookahead,
                       util_ceiling, dept_targets=None, budget=None, objective="pass_rate"):
    rows = {}
    for idx, r in sdf.iterrows():
        closed = r.is_open == 0
        rows[idx] = dict(
            idx=idx, dept=int(r.dept_id), store=int(r.store_id), closed=closed,
            reg=int(r.available_regular_employees), flex=int(r.cross_trained_flex_employees),
            ot=int(r.max_overtime_employees), hours=float(r.paid_hours), min=0 if closed else int(r.min_staff),
            r_reg=float(r.loaded_regular_hourly_cost), r_ot=float(r.loaded_overtime_hourly_cost),
            need=float(r.hours_needed), pc=float(r.p_callout),
            key=(r.store_id, r.day_date, r.shift_code),
        )
    # P(meet) table per block; value table v = what the optimizer maximizes (P(meet) by default)
    for r in rows.values():
        top = r["reg"] + r["flex"] + r["ot"]
        r["p"] = np.array([0.0] + [expected_service(model, r["dept"], r["store"], r["need"], r["hours"], n, r["pc"], backfill)[0]
                                   for n in range(1, top + 1)]) if not r["closed"] else np.zeros(1)
        if objective == "wait_time" and not r["closed"]:
            tgt = model.targets[r["dept"]]
            r["v"] = np.array([-WAIT_CAP] + [-expected_wait_ratio(model, r["dept"], r["store"], r["need"], r["hours"], n,
                                                                   r["pc"], backfill, tgt)
                                             for n in range(1, top + 1)])
        else:
            r["v"] = r["p"]
    # cost units
    units, unit_of = [], {}
    pairs = {}
    for r in rows.values():
        if r["dept"] in (FE, MS):
            pairs.setdefault(r["key"], {})[r["dept"]] = r
        else:
            units.append(_Unit([r]))
    for d in pairs.values():
        units.append(_Unit([d[FE], d[MS]]))
    for u in units:
        for j, r in enumerate(u.rows):
            unit_of[r["idx"]] = (u, j)

    n = {i: r["min"] for i, r in rows.items()}
    floor = {}  # filled after Phase 0/1
    open_rows = [i for i, r in rows.items() if not r["closed"]]

    def p_of(i, k):
        p = rows[i]["p"]
        return p[min(k, len(p) - 1)]

    def v_of(i, k):
        v = rows[i]["v"]
        return v[min(k, len(v) - 1)]

    def move(i, k, own_only=False):
        """Add k people to block i. If the partner department was borrowing this block's spare regular
        staff, the partner gives those people back (its headcount drops to its new capacity).
        own_only=True forbids borrowing from the partner (used in the first pass so every department
        claims its own people before anyone lends them out).
        Returns (value gain incl. partner loss, cost delta for ranking, changes, raw cost delta) or None."""
        u, j = unit_of[i]
        r = rows[i]
        ns = [n[x["idx"]] for x in u.rows]
        if ns[j] + k >= len(r["p"]):
            return None
        if own_only and ns[j] + k > r["reg"] + r["ot"]:
            return None
        c0, _ = u.cost_ok(ns)
        ns2 = list(ns)
        ns2[j] += k
        changes = {i: k}
        gain = v_of(i, ns2[j]) - v_of(i, ns[j])
        if len(u.rows) == 2:
            q = 1 - j
            pr, pi = u.rows[q], u.rows[q]["idx"]
            spare_from_i = max(r["reg"] - min(ns2[j], r["reg"]), 0)
            cap_p = pr["reg"] + pr["ot"] + min(pr["flex"], spare_from_i)
            if ns2[q] > cap_p:
                if cap_p < max(pr["min"], floor.get(pi, 0)):
                    return None  # never take people away from a shift below its safe (Phase 0) staffing
                changes[pi] = cap_p - ns2[q]
                gain += v_of(pi, cap_p) - v_of(pi, ns2[q])
                ns2[q] = cap_p
        c1, ok = u.cost_ok(ns2)
        if not ok:
            return None
        return gain, max(c1 - c0, 1e-6), changes, c1 - c0

    def apply(changes):
        for k, v in changes.items():
            n[k] += v

    def util_at(i, k):
        r = rows[i]
        worked = max(k * (1 - r["pc"] * (1 - backfill)), 1e-6)
        return r["need"] / (worked * r["hours"])

    # Phase 0: never abandon a block. Raise every block until projected utilization is at or below the
    # ceiling, or it runs out of people. Pass A uses each department's own regular + overtime staff only;
    # pass B then lets Front End and Merchandising borrow whatever the partner truly has spare.
    for own_only in (True, False):
        for i in open_rows:
            guard = 0
            while util_at(i, n[i]) > util_ceiling and guard < 200:
                guard += 1
                m = move(i, 1, own_only=own_only)
                if m is None or len(m[2]) > 1:
                    break
                apply(m[2])

    # Phase 1 (optional): raise every block to a service-probability floor
    for i in open_rows:
        guard = 0
        while p_of(i, n[i]) < block_floor and guard < max_extra:
            guard += 1
            m = move(i, 1)
            if m is None or m[0] <= 0:
                break
            apply(m[2])

    # Lock in Phase 0/1 staffing as a floor: later moves may add people anywhere, but may not pull a
    # borrowed person back out of a shift that already needed them to stay safe.
    floor.update(n)

    def total_cost():
        return sum(u.cost_ok([n[x["idx"]] for x in u.rows])[0] for u in units)

    remaining = None
    if budget is not None:
        spent = total_cost()
        if spent > budget:
            return pd.Series(n), round(spent, 2)  # minimum safe staffing already exceeds the budget
        remaining = budget - spent

    # Phase 2: greedy by value gained per dollar until the stop rule is met.
    # Each candidate looks ahead several people, because service vs staffing is S-shaped: one extra
    # person can look useless on a badly short block while three fix it.
    def best_move(i, max_cost=None):
        best = None
        for k in range(1, len(rows[i]["p"]) - n[i]):
            m = move(i, k)
            if m is None:
                break
            if max_cost is not None and m[3] > max_cost:
                continue
            if m[0] >= min_gain and (best is None or m[0] / m[1] > best[0] / best[1]):
                best = m
        return best

    # stop rule and eligibility
    dept_of = {i: rows[i]["dept"] for i in open_rows}
    if dept_targets:
        dept_n = {d: sum(1 for i in open_rows if dept_of[i] == d) for d in set(dept_of.values())}
        dept_total = {d: sum(p_of(i, n[i]) for i in open_rows if dept_of[i] == d) for d in dept_n}

        def below(d):
            return dept_total[d] < dept_targets.get(d, portfolio_target) * dept_n[d]

    total = sum(p_of(i, n[i]) for i in open_rows)
    target_total = portfolio_target * len(open_rows)

    def keep_going():
        if remaining is not None:
            return True  # budget mode: spend until no affordable move is left
        if dept_targets:
            return any(below(d) for d in dept_total)
        return total < target_total

    def eligible(i):
        if remaining is not None or not dept_targets:
            return True
        return below(dept_of[i])

    cache = {i: best_move(i) for i in open_rows}
    while keep_going():
        best = None
        for i, m in cache.items():
            if remaining is not None and m is not None and m[3] > remaining:
                m = best_move(i, max_cost=remaining)  # cached move unaffordable; look for a smaller one
            if m and eligible(i) and (best is None or m[0] / m[1] > best[0] / best[1]):
                best = m
        if best is None:
            break
        before = {k: p_of(k, n[k]) for k in best[2]}
        apply(best[2])
        if objective == "wait_time" or dept_targets:
            for k in best[2]:
                d = p_of(k, n[k]) - before[k]
                total += d
                if dept_targets and k in dept_of:
                    dept_total[dept_of[k]] += d
        else:
            total += best[0]
        if remaining is not None:
            remaining -= best[3]
        touched = {x["idx"] for k in best[2] for x in unit_of[k][0].rows}
        for k in touched:
            if k in cache:
                cache[k] = best_move(k)
    return pd.Series(n), None


def frontier(df: pd.DataFrame, model: ServiceModel, backfill: float, targets=None, block_floor: float = 0.0,
             util_ceiling: float = 0.90, **kw):
    """Cost vs expected % of blocks meeting target, for a range of portfolio targets (one scenario)."""
    from .costing import cost_plan
    targets = targets or [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]
    pts = []
    for t in targets:
        rec = optimize(df, model, backfill, t, block_floor, util_ceiling=util_ceiling, **kw)
        tmp = df.assign(n_staff=rec.values)
        tmp = cost_plan(tmp, "n_staff", "f")
        p = [expected_service(model, r.dept_id, r.store_id, r.hours_needed, r.paid_hours, int(r.n_staff), r.p_callout, backfill)[0]
             for r in tmp.itertuples() if r.is_open == 1]
        pts.append({"portfolio_target": t, "cost": tmp.f_cost.sum(), "expected_pct_meeting": float(np.mean(p))})
    return pd.DataFrame(pts)
