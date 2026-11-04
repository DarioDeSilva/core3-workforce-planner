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
from engine.optimizer import frontier
from engine.service import ServiceModel
from engine.staffing import build_plan, prepare, summarize
from engine.submission import make_submission, validate

st.set_page_config(page_title="Core 3 Staffing Planner", page_icon="📋", layout="wide")

DEPTS = {38: "Front End", 92: "Merchandising", 95: "Pharmacy"}
WAREHOUSES = {1: "Warehouse 1 (Medium)", 2: "Warehouse 2 (Large)", 4: "Warehouse 4 (Large)"}
SCENARIOS = {"BASE": "Baseline week", "PEAK": "Peak: promotion + winter storm", "DISRUPT": "Disruption: illness + crew shortage"}
SHIFTS = ["OPEN", "MID", "EVE"]
METRIC_LABEL = {38: "checkout wait", 92: "replenishment time", 95: "prescription wait"}
RISK_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828", "Closed": "#9e9e9e"}

PLAYBOOK = {
    "High": [
        "Pull cross-trained staff from the partner department if its block is low-risk.",
        "Offer overtime to the next shift's crew to start early or stay late.",
        "Escalate to the GM: borrow from a nearby warehouse or defer non-urgent tasks (e.g., stocking resets).",
    ],
    "Medium": [
        "Pre-approve one overtime backfill so any callout is replaced immediately.",
        "Check traffic at the start of the shift; if it runs >10% above forecast, apply the High-risk actions.",
    ],
}


# ------------------------------------------------------------------ cached engine calls
@st.cache_resource(show_spinner=False)
def service_model() -> ServiceModel:
    return ServiceModel.fit()


@st.cache_data(show_spinner=False)
def run_plan(params_key: str, params: dict, mode: str) -> pd.DataFrame:
    prepared = prepare(params)
    return build_plan(params, mode=mode, model=service_model(), prepared=prepared)


@st.cache_data(show_spinner=False)
def run_frontier(params_key: str, params: dict, scenario: str) -> pd.DataFrame:
    prepared = prepare(params)
    backfill = float(params["staffing"]["backfill_rate"])
    return frontier(prepared[prepared.scenario_code == scenario], service_model(), backfill,
                    util_ceiling=float(params["staffing"].get("utilization_ceiling", 0.9)))


def money(x: float) -> str:
    return f"${x:,.0f}"


def money_md(x: float) -> str:
    """Dollar amount safe inside Markdown (a bare $ pair renders as math)."""
    return money(x).replace("$", "\\$")


# ------------------------------------------------------------------ sidebar: inputs
base_params = data.load_params()
st.sidebar.title("Planner inputs")
scenario = st.sidebar.radio("Scenario", list(SCENARIOS), format_func=lambda s: f"{s} - {SCENARIOS[s]}")
mode_label = st.sidebar.radio("Planning method", ["Optimizer (recommended)", "Rule of thumb"],
                              help="Optimizer: cheapest plan that reaches the service goal. Rule of thumb: staff every block to a fixed utilization.")
mode = "optimizer" if mode_label.startswith("Optimizer") else "rule"

params = copy.deepcopy(base_params)
sp = params["staffing"]
if mode == "optimizer":
    sp["portfolio_service_target"] = st.sidebar.slider(
        "Service goal: share of shifts meeting target", 0.80, 0.99, float(sp["portfolio_service_target"]), 0.01,
        help="The optimizer adds staff where each dollar buys the most service until this goal is reached.")
else:
    u = st.sidebar.slider("Target utilization", 0.70, 0.88, 0.80, 0.01,
                          help="How busy staff should be. Service collapses above ~0.85.")
    sp["target_utilization"] = {38: u, 92: u, 95: u}

with st.sidebar.expander("Assumptions"):
    sp["utilization_ceiling"] = st.slider("Never let a shift exceed utilization", 0.80, 0.95,
                                          float(sp.get("utilization_ceiling", 0.90)), 0.01)
    sp["backfill_rate"] = st.slider("Share of callouts backfilled", 0.0, 1.0, float(sp["backfill_rate"]), 0.05)
    v = st.slider("Valentine's Day extra traffic (Front End/Merch)", 0.95, 1.15, 1.00, 0.01,
                  help="No historical analog; judgment call. 1.00 = no effect beyond the holiday weekend lift.")
    params["forecast"]["valentines_extra"] = {38: v, 92: v, 95: 1.0}

st.sidebar.divider()
wh = st.sidebar.selectbox("Warehouse", list(WAREHOUSES), format_func=WAREHOUSES.get)
dept = st.sidebar.selectbox("Department", list(DEPTS), format_func=DEPTS.get)

params_key = repr(params) + mode
with st.spinner("Planning the week..."):
    plan = run_plan(params_key, params, mode)

sc = plan[plan.scenario_code == scenario]
summ = summarize(plan).loc[scenario]

# ------------------------------------------------------------------ header + KPIs
st.title("Warehouse Staffing Planner")
st.caption(f"Planning week Thu Feb 12 - Wed Feb 18, 2026 · {SCENARIOS[scenario]} · all 3 warehouses · "
           "simulated case data (Texas MISA Fall 2026)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Labor cost (week)", money(summ.rec_cost), f"{money(summ.cost_change)} vs draft", delta_color="inverse")
k2.metric("Shifts expected to meet service target", f"{summ.rec_pct_meeting:.0%}",
          f"{(summ.rec_pct_meeting - summ.draft_pct_meeting) * 100:+.0f} pts vs draft")
k3.metric("High-risk shifts", int(summ.rec_high_risk), f"{int(summ.rec_high_risk - summ.draft_high_risk)} vs draft",
          delta_color="inverse")
k4.metric("Overtime / cross-trained people", f"{int(summ.rec_ot_people)} / {int(summ.rec_flex_people)}")
k5.metric("Infeasible shifts", int(summ.rec_infeasible), f"draft: {int(summ.draft_infeasible)}", delta_color="off")

tab_plan, tab_compare, tab_risk, tab_block, tab_frontier, tab_method = st.tabs(
    ["Staffing plan", "Plan vs draft", "Service risk & actions", "Explain a shift", "Cost vs service", "Method"])

# ------------------------------------------------------------------ tab: staffing plan heatmap
with tab_plan:
    v = sc[(sc.store_id == wh) & (sc.dept_id == dept)].copy()
    v["label"] = v.day_date.str[5:] + " " + v.day_name.str[:3]
    days = v.drop_duplicates("day_date").sort_values("day_date").label.tolist()
    st.subheader(f"{WAREHOUSES[wh]} · {DEPTS[dept]} · recommended employees per shift")
    staff = v.pivot(index="shift_code", columns="label", values="recommended_employees").reindex(SHIFTS)[days]
    draft = v.pivot(index="shift_code", columns="label", values="draft_scheduled_employees").reindex(SHIFTS)[days]
    risk = v.pivot(index="shift_code", columns="label", values="rec_risk").reindex(SHIFTS)[days]
    pm = v.pivot(index="shift_code", columns="label", values="rec_p_meet").reindex(SHIFTS)[days]
    text = [[("closed" if risk.iloc[i, j] == "Closed" else f"<b>{int(staff.iloc[i, j])}</b><br>draft {int(draft.iloc[i, j])}")
             for j in range(len(days))] for i in range(3)]
    fig = go.Figure(go.Heatmap(z=pm.fillna(1.05).values, x=days, y=SHIFTS, text=text, texttemplate="%{text}",
                               colorscale=[[0, "#c62828"], [0.6, "#f9a825"], [0.9, "#c8e6c9"], [0.95, "#2e7d32"], [1.0, "#e0e0e0"]],
                               zmin=0, zmax=1.05, showscale=False,
                               hovertemplate="%{x} %{y}<br>%{text}<extra></extra>"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Color = chance the shift meets its service target (green high, red low). Bold = our recommendation.")

    show = v[["label", "shift_code", "forecast_units", "recommended_employees", "draft_scheduled_employees",
              "rec_regular", "rec_flex", "rec_ot", "rec_cost", "rec_p_meet", "rec_expected_metric", "rec_risk"]].rename(columns={
        "label": "Day", "shift_code": "Shift", "forecast_units": "Forecast demand", "recommended_employees": "Recommended",
        "draft_scheduled_employees": "Draft", "rec_regular": "Regular", "rec_flex": "Cross-trained", "rec_ot": "Overtime",
        "rec_cost": "Cost", "rec_p_meet": "P(meet target)", "rec_expected_metric": f"Expected {METRIC_LABEL[dept]} (min)",
        "rec_risk": "Risk"})
    st.dataframe(show.style.format({"Forecast demand": "{:.1f}", "Cost": "${:,.0f}", "P(meet target)": "{:.0%}",
                                    f"Expected {METRIC_LABEL[dept]} (min)": "{:.1f}"}, na_rep="-"),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab: plan vs draft
with tab_compare:
    o = sc[sc.is_open == 1]
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
    f2 = px.bar(lp, x="Area", y="Share", color="Plan", barmode="group", title="Share of shifts expected to meet target",
                color_discrete_map={"Current draft": "#9e9e9e", "Our plan": "#1565c0"})
    f2.update_yaxes(tickformat=".0%", range=[0, 1])
    c2.plotly_chart(f2, width="stretch")

    st.subheader("Biggest changes from the draft")
    ch = o.assign(abs_change=o.change_vs_draft.abs()).sort_values("abs_change", ascending=False).head(15)
    st.dataframe(ch[["store_id", "department_group", "day_date", "shift_code", "draft_scheduled_employees",
                     "recommended_employees", "change_vs_draft", "draft_p_meet", "rec_p_meet", "why"]].rename(columns={
        "store_id": "WH", "department_group": "Dept", "day_date": "Date", "shift_code": "Shift",
        "draft_scheduled_employees": "Draft", "recommended_employees": "Ours", "change_vs_draft": "Change",
        "draft_p_meet": "Draft P(meet)", "rec_p_meet": "Our P(meet)", "why": "Why"}).style.format(
        {"Draft P(meet)": "{:.0%}", "Our P(meet)": "{:.0%}"}), hide_index=True, width="stretch")

# ------------------------------------------------------------------ tab: risk + actions
with tab_risk:
    r = sc[sc.rec_risk.isin(["High", "Medium"])].sort_values("rec_p_meet")
    if r.empty:
        st.success("Every open shift has at least a 90% chance of meeting its service target under this plan.")
    else:
        at_cap = int((r.recommended_employees >= r.rec_capacity).sum())
        st.write(f"**{len(r)} shifts** still carry service risk under our plan; {at_cap} of them are at capacity "
                 "(every available regular, cross-trained and overtime employee is already scheduled).")
        for lvl in ["High", "Medium"]:
            rr = r[r.rec_risk == lvl]
            if rr.empty:
                continue
            st.markdown(f"#### {lvl} risk ({len(rr)})")
            st.markdown("**What the warehouse leader should do:**\n" + "\n".join(f"- {a}" for a in PLAYBOOK[lvl]))
            st.dataframe(rr[["store_id", "department_group", "day_date", "shift_code", "recommended_employees",
                             "rec_capacity", "rec_p_meet", "rec_expected_metric", "service_target_value", "availability_note"]].rename(columns={
                "store_id": "WH", "department_group": "Dept", "day_date": "Date", "shift_code": "Shift",
                "recommended_employees": "Staff", "rec_capacity": "Capacity", "rec_p_meet": "P(meet)",
                "rec_expected_metric": "Expected minutes", "service_target_value": "Target minutes",
                "availability_note": "Availability note"}).style.format({"P(meet)": "{:.0%}", "Expected minutes": "{:.0f}"}),
                hide_index=True, width="stretch")

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
    bb.write(f"Regular {int(b.rec_regular)} · cross-trained {int(b.rec_flex)} · overtime {int(b.rec_ot)}")
    bb.write(f"Available: {int(b.available_regular_employees)} regular, {int(b.cross_trained_flex_employees)} cross-trained, "
             f"{int(b.max_overtime_employees)} overtime")
    bb.write(f"Expected callouts: {b.p_callout:.1%} of scheduled")
    if isinstance(b.availability_note, str):
        bb.info(b.availability_note)
    c.markdown("**3. Cost and service**")
    c.write(f"Cost: {money_md(b.rec_cost)} (draft {money_md(b.draft_cost)})")
    c.write(f"Projected utilization: {b.rec_utilization:.2f}")
    c.write(f"Expected {METRIC_LABEL[dept]}: {b.rec_expected_metric:.1f} min (target {b.service_target_value:.0f})")
    c.write(f"Chance of meeting target: **{b.rec_p_meet:.0%}** (draft {b.draft_p_meet:.0%})")
    st.markdown(f"**Why:** {b.why}")

    # service curve for this block
    model = service_model()
    from engine.staffing_core import expected_service
    ns = list(range(int(b.min_staff), int(max(b.rec_capacity, b.recommended_employees, b.draft_scheduled_employees)) + 4))
    ps = [expected_service(model, b.dept_id, b.store_id, b.hours_needed, b.paid_hours, n, b.p_callout,
                           float(params["staffing"]["backfill_rate"]))[0] for n in ns]
    f = px.line(x=ns, y=ps, markers=True, labels={"x": "Employees scheduled", "y": "Chance of meeting target"},
                title="How the chance of meeting target changes with staffing")
    f.add_vline(x=b.recommended_employees, line_color="#1565c0", annotation_text="ours")
    f.add_vline(x=b.draft_scheduled_employees, line_color="#9e9e9e", line_dash="dash", annotation_text="draft")
    f.add_vline(x=b.rec_capacity, line_color="#c62828", line_dash="dot", annotation_text="capacity")
    f.update_yaxes(tickformat=".0%", range=[0, 1.02])
    st.plotly_chart(f, width="stretch")

# ------------------------------------------------------------------ tab: frontier
with tab_frontier:
    st.write("Each point is the cheapest plan the optimizer can find for a given service goal. "
             "The gray marker is the current draft schedule.")
    if st.button("Compute cost vs service curve (~5 s)"):
        fr = run_frontier(params_key, params, scenario)
        f = px.line(fr, x="cost", y="expected_pct_meeting", markers=True,
                    labels={"cost": "Weekly labor cost", "expected_pct_meeting": "Share of shifts meeting target"})
        f.add_scatter(x=[summ.draft_cost], y=[summ.draft_pct_meeting], mode="markers+text", text=["current draft"],
                      textposition="top center", marker=dict(size=12, color="#9e9e9e"), name="Draft")
        f.update_yaxes(tickformat=".0%")
        f.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(f, width="stretch")
        st.dataframe(fr.style.format({"cost": "${:,.0f}", "expected_pct_meeting": "{:.1%}", "portfolio_target": "{:.0%}"}),
                     hide_index=True)

# ------------------------------------------------------------------ tab: method
with tab_method:
    st.markdown("""
**1. Forecast.** Normal-day average for each weekday (holiday windows excluded) x holiday lifts learned from the
MLK weekend (Front End +19%, Merchandising +15% on the pre-holiday weekend; Pharmacy -26% on the holiday) x the
scenario multiplier. Split into shifts with historical shift shares. Backtest on held-out weeks: about 40% less error than the naive benchmark.

**2. Service model.** From 1,000+ historical shifts: the chance a shift meets its target, given how busy staff are
(utilization) and the warehouse. Service holds up to ~0.80 utilization and collapses above ~0.85.

**3. Staffing.** Every shift is first staffed so utilization stays under the ceiling (or to capacity). Then the
optimizer adds people one decision at a time, always where each dollar buys the most service, until the goal is met.
Callouts are simulated (historical rates; about half backfilled with overtime).

**4. Cost.** Official rule: regular staff first, then cross-trained staff the partner department is not using,
then overtime (about 32% more per hour).
""")

# ------------------------------------------------------------------ downloads
st.sidebar.divider()
st.sidebar.markdown("**Downloads** (all three scenarios, current settings)")
sub = make_submission(plan, params["team_id"])
problems = validate(sub, plan)
st.sidebar.download_button("Submission CSV", sub.to_csv(index=False), "Core3Inc_submission.csv", "text/csv")
st.sidebar.download_button("Full plan CSV", plan.to_csv(index=False), "plan_by_block.csv", "text/csv")
if problems:
    st.sidebar.error("Submission issues: " + "; ".join(problems))
else:
    st.sidebar.success("Submission valid: 567 rows, all shifts feasible")
