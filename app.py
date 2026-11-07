"""Core 3 Inc. - Warehouse Staffing Planner (Streamlit proof of concept).

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine import data
from engine.compare import diff_scenarios, summarize_diff
from engine.forecast import backtest
from engine.optimizer import frontier
from engine.replan import replan
from engine.schedule import posting_schedule, schedule_html
from engine.service import ServiceModel
from engine.staffing import build_plan, prepare, summarize
from engine.staffing_core import expected_service
from engine.submission import make_submission, validate

st.set_page_config(page_title="Core 3 Staffing Planner", page_icon="📋", layout="wide")

DEPTS = {38: "Front End", 92: "Merchandising", 95: "Pharmacy"}
WAREHOUSES = {1: "Warehouse 1 (Medium)", 2: "Warehouse 2 (Large)", 4: "Warehouse 4 (Large)"}
SCENARIOS = {"BASE": "Baseline week", "PEAK": "Peak: promotion + winter storm", "DISRUPT": "Disruption: illness + crew shortage"}
SHIFTS = ["OPEN", "MID", "EVE"]
METRIC_LABEL = {38: "checkout wait", 92: "restocking time", 95: "prescription wait"}
METHODS = {"Optimizer (recommended)": "optimizer", "Budget": "budget", "Rule of thumb": "rule"}

# Every table header the user sees comes from here, so wording stays consistent.
LABELS = {
    "label": "Day", "store_id": "Warehouse", "department_group": "Department", "department": "Department",
    "day_date": "Date", "day_name": "Day", "shift_code": "Shift",
    "forecast_units": "Forecast demand (units)", "recommended_employees": "Our staff",
    "draft_scheduled_employees": "Draft staff", "rec_regular": "Regular", "rec_flex": "Borrowed (cross-trained)",
    "rec_ot": "Overtime", "rec_cost": "Cost", "rec_p_meet": "Chance of hitting target",
    "draft_p_meet": "Draft's chance", "rec_expected_metric": "Expected minutes", "rec_risk": "Risk",
    "rec_capacity": "Max staff available", "service_target_value": "Target minutes",
    "availability_note": "Availability note", "change_vs_draft": "Change vs draft", "why": "Why",
    "before": "Before", "after": "After", "change": "Change", "d_cost": "Cost change",
    "p_before": "Chance before", "p_after": "Chance after", "reason": "Why", "action": "Action",
}
PCT_COLS = ["Chance of hitting target", "Draft's chance", "Chance before", "Chance after"]

PLAYBOOK = {
    "High": [
        "Pull cross-trained staff from the partner department if its shift is low-risk.",
        "Offer overtime to the next shift's crew to start early or stay late.",
        "Escalate to the GM: borrow from a nearby warehouse or defer non-urgent tasks (e.g., stocking resets).",
    ],
    "Medium": [
        "Pre-approve one overtime backfill so any callout is replaced immediately.",
        "Check traffic at the start of the shift; if it runs >10% above forecast, use the Re-plan tab.",
    ],
}

GLOSSARY = """
- **Utilization**: how busy staff are = hours of work needed ÷ hours staff are paid. History shows service holds
  up to about 0.80 and collapses above 0.85.
- **Chance of hitting target**: probability the shift meets its service target (checkout wait ≤ 5 min, restocking
  ≤ 60 min, prescription wait ≤ 20 min), after simulating callouts.
- **Max staff available**: regular staff + cross-trained staff the partner department isn't using + overtime.
- **Borrowed (cross-trained)**: Front End and Merchandising staff can cover each other at the regular rate.
  Pharmacy can't (licensing).
- **Backfill**: a callout replaced by someone working overtime (about half are, historically).
- **Draft**: the managers' current schedule for the week, which repeats the usual pattern and ignores holidays.
"""


def table(df: pd.DataFrame, fmt: dict | None = None):
    """Show a dataframe with plain-language headers and standard formats."""
    out = df.rename(columns=LABELS)
    f = {c: "{:.0%}" for c in PCT_COLS if c in out}
    for c in ("Cost", "Cost change"):
        if c in out:
            f[c] = money
    if "Expected minutes" in out:
        f["Expected minutes"] = "{:.1f}"
    if "Forecast demand (units)" in out:
        f["Forecast demand (units)"] = "{:.1f}"
    f.update(fmt or {})
    st.dataframe(out.style.format(f, na_rep="-"), hide_index=True, width="stretch")


# ------------------------------------------------------------------ cached engine calls
@st.cache_resource(show_spinner=False)
def service_model() -> ServiceModel:
    return ServiceModel.fit()


@st.cache_resource(show_spinner=False)
def precomputed() -> dict | None:
    """Plan saved by scripts/run_plan.py for the default settings (instant first load)."""
    try:
        return pd.read_pickle(data.APP_CACHE)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def run_plan(key: str, params: dict, mode: str) -> tuple[pd.DataFrame, str]:
    pre = precomputed()
    if pre is not None and pre.get("key") == key:
        return pre["plan"], "precomputed"
    return build_plan(params, mode=mode, model=service_model(), prepared=prepare(params)), "live"


@st.cache_data(show_spinner=False)
def run_frontier(key: str, params: dict, scenario: str, use_saved: bool) -> pd.DataFrame:
    if use_saved:
        try:
            fr = pd.read_csv(data.OUTPUTS / "cost_service_frontier.csv")
            return fr[fr.scenario_code == scenario].drop(columns="scenario_code")
        except Exception:
            pass
    prepared = prepare(params)
    sp = params["staffing"]
    return frontier(prepared[prepared.scenario_code == scenario], service_model(), float(sp["backfill_rate"]),
                    util_ceiling=float(sp.get("utilization_ceiling", 0.9)), objective=sp.get("objective", "pass_rate"))


@st.cache_data(show_spinner=False)
def run_backtest() -> pd.DataFrame:
    return backtest().reset_index()


@st.cache_data(show_spinner=False)
def run_replan(key: str, params: dict, scenario: str, as_of: str, store: int, dept, traffic: float, callouts: int,
               mode: str):
    return replan(params, service_model(), scenario, as_of, store, dept, traffic, callouts, mode)


def money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def money_md(x: float) -> str:
    """Dollar amount safe inside Markdown (a bare $ pair renders as math)."""
    return money(x).replace("$", "\\$")


# light palette so dark cell text is readable in both light and dark themes
COLORSCALE = [[0, "#ef9a9a"], [0.6 / 1.05, "#ffe082"], [0.9 / 1.05, "#dcedc8"], [0.99 / 1.05, "#a5d6a7"],
              [1.0 / 1.05, "#81c784"], [1.0, "#e0e0e0"]]


def risk_heatmap(z: pd.DataFrame, text: list[list[str]], height: int):
    """Heatmap colored by chance of hitting target (NaN = closed, gray), dark text on every cell."""
    zz = z.fillna(1.05).values
    fig = go.Figure(go.Heatmap(z=zz, x=list(z.columns), y=list(z.index), colorscale=COLORSCALE, zmin=0, zmax=1.05,
                               showscale=False, xgap=2, ygap=2, customdata=text,
                               hovertemplate="%{x} · %{y}<br>%{customdata}<extra></extra>"))
    for i, row in enumerate(z.index):
        for j, col in enumerate(z.columns):
            fig.add_annotation(x=col, y=row, text=text[i][j], showarrow=False, font=dict(color="#1b1b1b", size=12))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def cell_text(r) -> str:
    if r.is_open == 0:
        return "closed"
    return f"<b>{int(r.recommended_employees)}</b><br>draft {int(r.draft_scheduled_employees)}"


# ------------------------------------------------------------------ sidebar: inputs
base_params = data.load_params()
params = copy.deepcopy(base_params)
sp = params["staffing"]
fp = params["forecast"]

st.sidebar.title("Planner inputs")
scenario = st.sidebar.radio("Scenario", list(SCENARIOS), format_func=lambda s: f"{s} - {SCENARIOS[s]}",
                            help="The planning week under expected conditions (BASE), a promotion + storm surge "
                                 "(PEAK), or illness and crew shortages (DISRUPT).")
mode = METHODS[st.sidebar.radio(
    "Planning method", list(METHODS),
    help="Optimizer: cheapest plan that reaches your service goal. Budget: best service you can buy with a fixed "
         "weekly budget. Rule of thumb: staff every shift to a fixed level of busyness.")]

if mode == "optimizer":
    by_dept = st.sidebar.checkbox("Set a different goal per department",
                                  help="E.g. hold Pharmacy and checkout to 99% but accept 95% for restocking.")
    if by_dept:
        goals = {d: st.sidebar.slider(f"{DEPTS[d]} goal", 0.80, 0.99, 0.99, 0.01) for d in DEPTS}
        sp["service_target_by_dept"] = goals
    else:
        sp["portfolio_service_target"] = st.sidebar.slider(
            "Service goal: share of shifts hitting target", 0.80, 0.99, float(sp["portfolio_service_target"]), 0.01,
            help="The optimizer adds staff where each dollar buys the most service until this goal is reached.")
elif mode == "budget":
    sp["budget"] = float(st.sidebar.number_input(
        "Weekly labor budget ($, all 3 warehouses)", min_value=200_000, max_value=500_000, value=350_000, step=5_000,
        help="The optimizer first makes every shift safe (never above the busyness ceiling), then spends the rest "
             "of the budget wherever it buys the most service. Applies to each scenario."))
else:
    u = st.sidebar.slider("Target busyness (utilization)", 0.70, 0.88, 0.80, 0.01,
                          help="How busy staff should be. Service collapses above ~0.85.")
    sp["target_utilization"] = {38: u, 92: u, 95: u}

with st.sidebar.expander("Assumptions"):
    sp["utilization_ceiling"] = st.slider(
        "Never let a shift get busier than", 0.80, 0.95, float(sp.get("utilization_ceiling", 0.90)), 0.01,
        help="Every shift is staffed at least until staff are this busy (or everyone available is scheduled).")
    sp["backfill_rate"] = st.slider("Share of callouts replaced with overtime", 0.0, 1.0, float(sp["backfill_rate"]),
                                    0.05, help="History: about half.")
    fp["holiday_lift_scale"] = st.slider(
        "Holiday effect strength (1.0 = same as MLK weekend)", 0.80, 1.20, float(fp.get("holiday_lift_scale", 1.0)),
        0.05, help="Presidents' Day is forecast from the MLK weekend. 0.8 = a 20% weaker holiday effect, 1.2 = stronger.")
    v = st.slider("Valentine's Day extra traffic (Front End/Merch)", 0.95, 1.15, 1.00, 0.01,
                  help="No historical analog; judgment call. 1.00 = no effect beyond the holiday-weekend lift.")
    fp["valentines_extra"] = {38: v, 92: v, 95: 1.0}
    obj = st.radio("Optimize for", ["Shifts hitting target", "Shorter average waits"],
                   help="Shifts hitting target: maximize how many shifts meet their service target. Shorter average "
                        "waits: reduce expected minutes vs target across all shifts (each shift capped at 3x target).")
    sp["objective"] = "pass_rate" if obj.startswith("Shifts") else "wait_time"

st.sidebar.divider()
wh = st.sidebar.selectbox("Warehouse", list(WAREHOUSES), format_func=WAREHOUSES.get)
dept = st.sidebar.selectbox("Department", list(DEPTS), format_func=DEPTS.get)

key = data.canonical_key(params, mode)
with st.spinner("Planning the week..."):
    plan, source = run_plan(key, params, mode)

sc = plan[plan.scenario_code == scenario]
summ = summarize(plan).loc[scenario]

# ------------------------------------------------------------------ header + KPIs
st.title("Warehouse Staffing Planner")
st.caption(f"Planning week Thu Feb 12 - Wed Feb 18, 2026 · {SCENARIOS[scenario]} · all 3 warehouses · "
           "simulated case data (Texas MISA Fall 2026)")

if mode == "budget" and "budget_min_cost" in sc and sc.budget_min_cost.notna().any():
    st.warning(f"Budget too low for {scenario}: making every shift safe already costs "
               f"{money_md(sc.budget_min_cost.dropna().iloc[0])}. Showing that minimum safe plan.")

k1, k2, k3, k4, k5 = st.columns(5)
cost_label = f"Labor cost (budget {money(sp['budget'])})" if mode == "budget" else "Labor cost (week)"
k1.metric(cost_label, money(summ.rec_cost), f"{money(summ.cost_change)} vs draft", delta_color="inverse",
          help="Total weekly labor cost for all 3 warehouses, using the official rule: regular staff first, then "
               "borrowed cross-trained staff, then overtime (~32% more per hour).")
k2.metric("Shifts expected to hit target", f"{summ.rec_pct_meeting:.0%}",
          f"{(summ.rec_pct_meeting - summ.draft_pct_meeting) * 100:+.0f} pts vs draft",
          help="Average chance that a shift meets its service target, after simulating callouts.")
k3.metric("High-risk shifts", int(summ.rec_high_risk), f"{int(summ.rec_high_risk - summ.draft_high_risk)} vs draft",
          delta_color="inverse", help="Shifts with less than a 60% chance of hitting target.")
k4.metric("Overtime / borrowed shifts", f"{int(summ.rec_ot_people)} / {int(summ.rec_flex_people)}",
          help="Person-shifts paid at overtime, and person-shifts covered by cross-trained staff from the partner "
               "department, across the week.")
k5.metric("Infeasible shifts", int(summ.rec_infeasible),
          help="Shifts scheduled above the staff actually available, or below the minimum staffing floor.")
k5.caption(f"Draft: {int(summ.draft_infeasible)}")

tabs = st.tabs(["Staffing plan", "What changed", "Re-plan", "Plan vs draft", "Service risk & actions",
                "Explain a shift", "Cost vs service", "Method"])
tab_plan, tab_changed, tab_replan, tab_compare, tab_risk, tab_block, tab_frontier, tab_method = tabs

# ------------------------------------------------------------------ tab: staffing plan heatmap
with tab_plan:
    view = st.radio("View", ["One department", "Whole warehouse"], horizontal=True, label_visibility="collapsed")
    w = sc[sc.store_id == wh].copy()
    w["label"] = w.day_date.str[5:] + " " + w.day_name.str[:3]
    days = w.drop_duplicates("day_date").sort_values("day_date").label.tolist()
    if view == "One department":
        v = w[w.dept_id == dept]
        st.subheader(f"{WAREHOUSES[wh]} · {DEPTS[dept]} · recommended staff per shift")
        z = v.pivot(index="shift_code", columns="label", values="rec_p_meet").reindex(SHIFTS)[days]
        cells = v.set_index(["shift_code", "label"])
        text = [[cell_text(cells.loc[(s, d)]) for d in days] for s in SHIFTS]
        st.plotly_chart(risk_heatmap(z, text, 280), width="stretch")
        st.caption("Color = chance the shift hits its service target (green high, amber medium, red low; gray closed). "
                   "Bold = our recommendation.")
        table(v[["label", "shift_code", "forecast_units", "recommended_employees", "draft_scheduled_employees",
                 "rec_regular", "rec_flex", "rec_ot", "rec_cost", "rec_p_meet", "rec_expected_metric", "rec_risk"]]
              .rename(columns={"rec_expected_metric": f"Expected {METRIC_LABEL[dept]} (min)"}),
              {f"Expected {METRIC_LABEL[dept]} (min)": "{:.1f}"})
    else:
        st.subheader(f"{WAREHOUSES[wh]} · every department · recommended staff per shift")
        w["row"] = w.dept_id.map(DEPTS) + " · " + w.shift_code
        rows = [f"{DEPTS[d]} · {s}" for d in DEPTS for s in SHIFTS]
        z = w.pivot(index="row", columns="label", values="rec_p_meet").reindex(rows)[days]
        cells = w.set_index(["row", "label"])
        text = [[cell_text(cells.loc[(r, d)]) for d in days] for r in rows]
        st.plotly_chart(risk_heatmap(z, text, 620), width="stretch")
        st.caption("Color = chance the shift hits its service target (green high, amber medium, red low; gray closed).")
        o = w[w.is_open == 1]
        ds = o.groupby("dept_id").agg(cost=("rec_cost", "sum"), draft=("draft_cost", "sum"), p=("rec_p_meet", "mean"),
                                      high=("rec_risk", lambda x: (x == "High").sum())).reset_index()
        ds["Department"] = ds.dept_id.map(DEPTS)
        st.dataframe(ds[["Department", "cost", "draft", "p", "high"]].rename(columns={
            "cost": "Week cost", "draft": "Draft cost", "p": "Shifts hitting target", "high": "High-risk shifts"})
                     .style.format({"Week cost": "${:,.0f}", "Draft cost": "${:,.0f}", "Shifts hitting target": "{:.0%}"}),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab: what changed between scenarios
with tab_changed:
    others = [s for s in SCENARIOS if s != scenario]
    ref = st.selectbox(f"Compare {scenario} with", others, index=0,
                       help="Shows what a warehouse leader must change to go from that plan to this one.")
    d = diff_scenarios(plan, ref, scenario)
    s_ = summarize_diff(d)
    st.markdown(f"**{ref} → {scenario}:** {s_['shifts']} shifts change headcount "
                f"(+{s_['added']} / -{s_['removed']} people, {s_['hours']:+,.0f} staff-hours), "
                f"labor cost {money_md(s_['cost'])}. {s_['direct']} of the changed shifts are directly affected by the "
                "scenario; the rest are knock-ons (cross-trained staff moving between Front End and Merchandising) or "
                "the optimizer rebalancing the week to keep the overall service goal.")
    c1, c2 = st.columns([1, 1])
    only_direct = c1.toggle("Only shifts directly affected", value=False)
    whf = c2.selectbox("Warehouse", ["All"] + list(WAREHOUSES), format_func=lambda x: x if x == "All" else WAREHOUSES[x],
                       key="chg_wh")
    dd = d[d.direct] if only_direct else d
    if whf != "All":
        dd = dd[dd.store_id == whf]
    if dd.empty:
        st.info("No changes for this selection.")
    else:
        table(dd[["action", "before", "after", "d_cost", "p_before", "p_after", "reason"]])
        st.download_button("Download action list (CSV)", dd.to_csv(index=False),
                           f"actions_{ref}_to_{scenario}.csv", "text/csv")

# ------------------------------------------------------------------ tab: mid-week re-plan
with tab_replan:
    st.write("Something changed mid-week? Enter what you're seeing; the planner keeps the shifts already worked and "
             "re-optimizes the rest of the week.")
    rp_mode = "optimizer" if mode == "budget" else mode
    with st.form("replan"):
        c1, c2, c3 = st.columns(3)
        dates = sorted(sc.day_date.unique())
        as_of = c1.selectbox("Starting from", dates, index=1, format_func=lambda x: f"{x} ({pd.Timestamp(x):%a})")
        r_wh = c2.selectbox("Warehouse", list(WAREHOUSES), format_func=WAREHOUSES.get, index=2, key="rp_wh")
        r_dept = c3.selectbox("Department", ["All"] + list(DEPTS), format_func=lambda x: x if x == "All" else DEPTS[x],
                              index=1, key="rp_dept")
        c4, c5 = st.columns(2)
        traffic = c4.slider("Traffic vs forecast", -30, 50, 15, 5, format="%d%%",
                            help="How far actual demand is running above (+) or below (-) the forecast.")
        callouts = c5.number_input("Extra callouts per shift", 0, 10, 0,
                                   help="Additional people out sick per shift, beyond normal callouts.")
        go_btn = st.form_submit_button("Re-plan", type="primary")
    if go_btn:
        st.session_state["rp"] = (scenario, as_of, r_wh, None if r_dept == "All" else r_dept, traffic / 100, int(callouts))
    if "rp" in st.session_state and st.session_state["rp"][0] == scenario:
        _, a_, w_, d_, t_, c_ = st.session_state["rp"]
        with st.spinner("Re-planning the rest of the week..."):
            rd, before, after, rs = run_replan(key, params, scenario, a_, w_, d_, t_, c_, rp_mode)
        where = f"{WAREHOUSES[w_]} · {'all departments' if d_ is None else DEPTS[d_]}"
        st.markdown(f"**{where}, from {a_}: traffic {t_:+.0%}, {c_} extra callouts per shift.**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Cost, rest of week", money(rs["cost_after"]), f"{money(rs['cost_after'] - rs['cost_before'])}",
                  delta_color="inverse")
        m2.metric("Shifts hitting target, rest of week", f"{rs['pass_after']:.0%}",
                  f"{(rs['pass_after'] - rs['pass_before']) * 100:+.1f} pts")
        m3.metric("Shifts to change", rs["shifts"], f"{rs['added']} added / {rs['removed']} removed", delta_color="off")
        if rs["infeasible_after"]:
            st.error(f"{rs['infeasible_after']} shifts can no longer be staffed to their minimum. Escalate to the GM: "
                     "borrow from another warehouse.")
        if rd.empty:
            st.success("No staffing changes needed.")
        else:
            table(rd[["action", "before", "after", "d_cost", "p_before", "p_after", "reason"]])
            capped = rd[rd.at_capacity & (rd.p_after < 0.9)]
            if len(capped):
                st.markdown(f"**{len(capped)} shifts are at max staff and still at risk.** "
                            + " ".join(PLAYBOOK["High"]))
        st.caption("Compared with re-planning the same remaining days without the change, so every difference is "
                   "caused by what you entered.")

# ------------------------------------------------------------------ tab: plan vs draft
with tab_compare:
    o = sc[sc.is_open == 1]
    w_ours = np.minimum(o.rec_expected_metric / o.service_target_value, 3).mean()
    w_draft = np.minimum(o.draft_expected_metric / o.service_target_value, 3).mean()
    st.metric("Average wait vs target (1.0 = exactly at target)", f"{w_ours:.2f}x", f"{w_ours - w_draft:+.2f}x vs draft",
              delta_color="inverse", help="Expected minutes ÷ target minutes, averaged over all open shifts "
                                          "(each capped at 3x). Lower is better.")
    c1, c2 = st.columns(2)
    by = o.groupby(["store_id", "dept_id"]).agg(rec=("rec_cost", "sum"), draft=("draft_cost", "sum"),
                                                rp=("rec_p_meet", "mean"), dp=("draft_p_meet", "mean")).reset_index()
    by["Area"] = "W" + by.store_id.astype(str) + " " + by.dept_id.map(DEPTS)
    long = by.melt(id_vars="Area", value_vars=["draft", "rec"], var_name="Plan", value_name="Cost")
    long["Plan"] = long.Plan.map({"draft": "Current draft", "rec": "Our plan"})
    c1.plotly_chart(px.bar(long, x="Area", y="Cost", color="Plan", barmode="group", title="Weekly labor cost",
                           color_discrete_map={"Current draft": "#9e9e9e", "Our plan": "#1565c0"}), width="stretch")
    lp = by.melt(id_vars="Area", value_vars=["dp", "rp"], var_name="Plan", value_name="Share")
    lp["Plan"] = lp.Plan.map({"dp": "Current draft", "rp": "Our plan"})
    f2 = px.bar(lp, x="Area", y="Share", color="Plan", barmode="group", title="Share of shifts expected to hit target",
                color_discrete_map={"Current draft": "#9e9e9e", "Our plan": "#1565c0"})
    f2.update_yaxes(tickformat=".0%", range=[0, 1])
    c2.plotly_chart(f2, width="stretch")

    st.subheader("Biggest changes from the draft")
    ch = o.assign(abs_change=o.change_vs_draft.abs()).sort_values("abs_change", ascending=False).head(15)
    table(ch[["store_id", "department_group", "day_date", "shift_code", "draft_scheduled_employees",
              "recommended_employees", "change_vs_draft", "draft_p_meet", "rec_p_meet", "why"]])

# ------------------------------------------------------------------ tab: risk + actions
with tab_risk:
    r = sc[sc.rec_risk.isin(["High", "Medium"])].sort_values("rec_p_meet")
    if r.empty:
        st.success("Every open shift has at least a 90% chance of hitting its service target under this plan.")
    else:
        at_cap = int((r.recommended_employees >= r.rec_capacity).sum())
        st.write(f"**{len(r)} shifts** still carry service risk under our plan; {at_cap} of them are at max staff "
                 "(every available regular, cross-trained and overtime employee is already scheduled).")
        for lvl in ["High", "Medium"]:
            rr = r[r.rec_risk == lvl]
            if rr.empty:
                continue
            st.markdown(f"#### {lvl} risk ({len(rr)})")
            st.markdown("**What the warehouse leader should do:**\n" + "\n".join(f"- {a}" for a in PLAYBOOK[lvl]))
            table(rr[["store_id", "department_group", "day_date", "shift_code", "recommended_employees",
                      "rec_capacity", "rec_p_meet", "rec_expected_metric", "service_target_value", "availability_note"]],
                  {"Expected minutes": "{:.0f}"})

# ------------------------------------------------------------------ tab: explain one block
with tab_block:
    v = sc[(sc.store_id == wh) & (sc.dept_id == dept) & (sc.is_open == 1)].copy()
    v["label"] = v.day_date + " " + v.day_name.str[:3] + " " + v.shift_code
    pick = st.selectbox("Shift", v.label.tolist())
    b = v[v.label == pick].iloc[0]
    st.subheader(f"{WAREHOUSES[wh]} · {DEPTS[dept]} · {pick}")
    a, bb, c = st.columns(3)
    a.markdown("**1. Demand forecast**")
    a.write(f"Normal {b.day_name} baseline: {b.baseline:.1f} units/day")
    a.write(f"Holiday lift: x{b.holiday_lift:.2f} ({b.holiday_window_role})")
    a.write(f"Share of day in this shift: {b.share:.0%}")
    a.write(f"Scenario multiplier: x{b.demand_multiplier:.2f}")
    a.write(f"**Forecast: {b.forecast_units:.1f} units = {b.hours_needed:.1f} hours of work**")
    bb.markdown("**2. Staffing**")
    bb.write(f"Recommended: **{int(b.recommended_employees)}** (draft {int(b.draft_scheduled_employees)}, minimum {int(b.min_staff)})")
    bb.write(f"Regular {int(b.rec_regular)} · borrowed {int(b.rec_flex)} · overtime {int(b.rec_ot)}")
    bb.write(f"Available: {int(b.available_regular_employees)} regular, {int(b.cross_trained_flex_employees)} cross-trained, "
             f"{int(b.max_overtime_employees)} overtime")
    bb.write(f"Expected callouts: {b.p_callout:.1%} of scheduled")
    if isinstance(b.availability_note, str):
        bb.info(b.availability_note)
    c.markdown("**3. Cost and service**")
    c.write(f"Cost: {money_md(b.rec_cost)} (draft {money_md(b.draft_cost)})")
    c.write(f"How busy staff will be: {b.rec_utilization:.0%} of paid time")
    c.write(f"Expected {METRIC_LABEL[dept]}: {b.rec_expected_metric:.1f} min (target {b.service_target_value:.0f})")
    c.write(f"Chance of hitting target: **{b.rec_p_meet:.0%}** (draft {b.draft_p_meet:.0%})")
    st.markdown(f"**Why:** {b.why}")

    ns = list(range(int(b.min_staff), int(max(b.rec_capacity, b.recommended_employees, b.draft_scheduled_employees)) + 4))
    ps = [expected_service(service_model(), b.dept_id, b.store_id, b.hours_needed, b.paid_hours, n, b.p_callout,
                           float(sp["backfill_rate"]))[0] for n in ns]
    f = px.line(x=ns, y=ps, markers=True, labels={"x": "Employees scheduled", "y": "Chance of hitting target"},
                title="How the chance of hitting target changes with staffing")
    at_capacity = round(b.recommended_employees) == round(b.rec_capacity)
    f.add_vline(x=b.recommended_employees, line_color="#1565c0",
                annotation_text="ours = max available" if at_capacity else "ours")
    f.add_vline(x=b.draft_scheduled_employees, line_color="#9e9e9e", line_dash="dash", annotation_text="draft")
    if not at_capacity:
        f.add_vline(x=b.rec_capacity, line_color="#c62828", line_dash="dot", annotation_text="max available")
    f.update_yaxes(tickformat=".0%", range=[0, 1.02])
    st.plotly_chart(f, width="stretch")

# ------------------------------------------------------------------ tab: frontier
with tab_frontier:
    st.write("Each point is the cheapest plan the optimizer can find for a given service goal. "
             "The gray marker is the current draft schedule.")
    default_settings = source == "precomputed"
    if default_settings or st.button("Compute cost vs service curve (~5 s)"):
        fr = run_frontier(key, params, scenario, default_settings)
        f = px.line(fr, x="cost", y="expected_pct_meeting", markers=True,
                    labels={"cost": "Weekly labor cost", "expected_pct_meeting": "Share of shifts hitting target"})
        f.add_scatter(x=[summ.draft_cost], y=[summ.draft_pct_meeting], mode="markers+text", text=["current draft"],
                      textposition="top center", marker=dict(size=12, color="#9e9e9e"), name="Draft")
        f.add_scatter(x=[summ.rec_cost], y=[summ.rec_pct_meeting], mode="markers+text", text=["this plan"],
                      textposition="bottom center", marker=dict(size=12, color="#1565c0", symbol="star"), name="This plan")
        f.update_yaxes(tickformat=".0%")
        f.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(f, width="stretch")
        st.dataframe(fr.rename(columns={"portfolio_target": "Service goal", "cost": "Cost",
                                        "expected_pct_meeting": "Shifts hitting target"})
                     .style.format({"Cost": "${:,.0f}", "Shifts hitting target": "{:.1%}", "Service goal": "{:.0%}"}),
                     hide_index=True)

# ------------------------------------------------------------------ tab: method
with tab_method:
    st.markdown("""
**1. Forecast.** Normal-day average for each weekday (holiday windows excluded) x holiday lifts learned from the
MLK weekend (Front End +19%, Merchandising +15% on the pre-holiday weekend; Pharmacy -26% on the holiday) x the
scenario multiplier. Split into shifts with historical shift shares.

**2. Service model.** From 1,000+ historical shifts: the chance a shift hits its target, given how busy staff are
(utilization) and the warehouse. Service holds up to ~0.80 utilization and collapses above ~0.85.

**3. Staffing.** Every shift is first staffed so staff are never busier than the ceiling (or to max available). Then
the optimizer adds people one decision at a time, always where each dollar buys the most service, until the goal is
met (or the budget is spent). Callouts are simulated (historical rates; about half backfilled with overtime).

**4. Cost.** Official rule: regular staff first, then cross-trained staff the partner department is not using,
then overtime (about 32% more per hour).

**5. Reacting to change.** *What changed* turns the difference between two scenarios into a list of actions.
*Re-plan* keeps the shifts already worked and re-optimizes the rest of the week for what's actually happening.
""")
    st.subheader("How accurate is the forecast?")
    bt = run_backtest()
    bt["Department"] = bt.dept_id.map(DEPTS)
    avg = bt.groupby("Department")[["mape_ours", "mape_naive"]].mean().reset_index().melt(
        id_vars="Department", var_name="Forecast", value_name="Average error")
    avg["Forecast"] = avg.Forecast.map({"mape_ours": "Our forecast", "mape_naive": "Naive benchmark"})
    fa = px.bar(avg, x="Department", y="Average error", color="Forecast", barmode="group", text_auto=".1%",
                color_discrete_map={"Our forecast": "#1565c0", "Naive benchmark": "#9e9e9e"})
    fa.update_yaxes(tickformat=".0%")
    st.plotly_chart(fa, width="stretch")
    st.caption(f"Shift-level error on held-out weeks 5-6 (trained only on earlier weeks): "
               f"ours {bt.mape_ours.mean():.1%} vs naive same-weekday average {bt.mape_naive.mean():.1%}.")
    st.subheader("Glossary")
    st.markdown(GLOSSARY)

# ------------------------------------------------------------------ downloads
st.sidebar.divider()
st.sidebar.markdown("**Downloads**")
sub = make_submission(plan, params["team_id"])
problems = validate(sub, plan)
st.sidebar.download_button("Submission CSV (all scenarios)", sub.to_csv(index=False), "Core3Inc_submission.csv", "text/csv")
st.sidebar.download_button("Full plan CSV (all scenarios)", plan.to_csv(index=False), "plan_by_block.csv", "text/csv")
sched = posting_schedule(sc, wh)
title = f"{WAREHOUSES[wh]} - staffing schedule, week of Feb 12, 2026"
st.sidebar.download_button(f"Printable schedule: W{wh}, {scenario}",
                           schedule_html(sched, title, f"Scenario: {scenario} ({SCENARIOS[scenario]}). "
                                                       "Print or save as PDF from your browser."),
                           f"schedule_W{wh}_{scenario}.html", "text/html")
st.sidebar.download_button(f"Schedule CSV: W{wh}, {scenario}", sched.to_csv(index=False),
                           f"schedule_W{wh}_{scenario}.csv", "text/csv")
if problems:
    st.sidebar.error("Submission issues: " + "; ".join(problems))
else:
    st.sidebar.success("Submission valid: 567 rows, all shifts feasible")
st.sidebar.caption("Plan source: " + ("precomputed (instant)" if source == "precomputed" else "computed live for these settings"))
