# Core 3 Inc. — Warehouse Staffing Planner

Texas MISA Fall 2026 Case Competition (Costco Workforce Planning). **All data is simulated for the competition; it is not Costco data.**

A proof-of-concept tool a warehouse leader can use to pick a scenario and see **how many people to schedule in every shift, what it costs, and where service is at risk**, plus *why* for each recommendation.

## Results so far (optimizer, service goal 99%)

| Scenario | Our cost | Draft cost | Shifts expected to meet target (ours / draft) | High-risk shifts (ours / draft) |
|---|---|---|---|---|
| BASE | $374,788 | $329,108 | 99% / 62% | 0 / 72 |
| PEAK | $393,382 | $329,108 | 90% / 55% | 19 / 81 |
| DISRUPT | $371,428 | $326,367 | 93% / 61% | 12 / 73 (draft has 12 infeasible shifts) |

On the cost-vs-service curve, BASE reaches 80% of shifts on target for ~$336K (about 2% above the draft) 98% for ~$364K (+11%), and 99% for ~$375K (+14%).
Remaining PEAK/DISRUPT risk is in shifts that are already at capacity. Numbers come from `outputs/`; rerun the engine after changing any assumption.

## What the tool does

| Tab | What a warehouse leader gets |
|---|---|
| Staffing plan | Recommended staff per shift vs the draft, colored by chance of hitting target. One department, or the **whole warehouse** (all departments × shifts × days). |
| What changed | The difference between two scenarios (e.g. BASE → PEAK) as a list of actions: "W4 Front End Fri 02-13 EVE: add 2 (+1 overtime, +1 borrowed from Merch), +$272". Filter to shifts directly affected. |
| Re-plan | Mid-week: enter traffic vs forecast and extra callouts; the shifts already worked stay locked and the rest of the week is re-optimized. |
| Plan vs draft | Cost and service by warehouse/department, average wait vs target, biggest changes with reasons. |
| Service risk & actions | Shifts still at risk and what to do about them. |
| Explain a shift | The full chain from forecast to staffing, cost and service, plus a chart of service vs headcount. |
| Cost vs service | Cheapest plan for each service goal, with the draft and this plan marked. |
| Method | How it works, forecast accuracy chart, glossary. |

Sidebar options: planning method (**Optimizer**, **Budget**: best service for a weekly budget, or **Rule of thumb**), service goal (one goal or **one per department**), and assumptions (busyness ceiling, backfill, **holiday effect strength**, Valentine's effect, **optimize for shifts hitting target or shorter average waits**). Downloads: submission CSV, full plan, and a **printable schedule** per warehouse (HTML → print to PDF, one page per day).

With default settings the app opens instantly from `outputs/app_cache.pkl` (written by `scripts/run_plan.py`); any change recomputes live.

## How it works

1. **Forecast** (`engine/forecast.py`): average of the same weekday over normal days, × holiday lifts learned from the MLK weekend, split into shifts with historical shift shares, × the scenario's demand multiplier. On held-out weeks the shift-level error is **4.6% vs 7.7%** for the naive benchmark.
2. **Service model** (`engine/service.py`): fitted on 1,000+ historical shifts, it gives the chance a shift meets its target given utilization and warehouse. Service holds to ~0.80 utilization and collapses above ~0.85.
3. **Staffing** (`engine/staffing.py`, `engine/optimizer.py`):
   - *Optimizer (recommended):* first staff every shift to at most 0.90 utilization (or capacity), then add people one decision at a time wherever each dollar buys the most service, until the service goal is met. Callouts are simulated from historical rates, with about half backfilled.
   - *Rule of thumb:* staff every shift to a target utilization plus a callout buffer.
4. **Cost and feasibility** (`engine/costing.py`): the official rule. Regular staff first, then cross-trained staff the partner department isn't using, then overtime.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_plan.py --frontier   # writes everything in outputs/
python scripts/backtest.py              # forecast accuracy check
streamlit run app.py                    # opens the tool at http://localhost:8501
```

## Who edits what

| Person | Works in | Notes |
|---|---|---|
| Leo (analysis) | `config/parameters.yaml`, `notebooks/`, `engine/forecast.py` | Change holiday lifts, Valentine's effect, utilization targets, callout rates. Rerun `scripts/run_plan.py`. |
| Mia (visuals) | `outputs/plan_by_block.csv`, `outputs/scenario_summary.csv`, `outputs/cost_service_frontier.csv` | Tableau sources. One row per scenario × warehouse × department × day × shift. Don't edit by hand; ask for a rerun. |
| Dario (tool) | `app.py`, `engine/` | Optimizer, app, submission. |

**Rule: every number in the deck comes from `outputs/`, so the deck, the tool and the submission always agree.**

### Key columns in `plan_by_block.csv`
`forecast_units` (demand), `recommended_employees` (our plan), `draft_scheduled_employees` (manager's draft), `rec_regular/rec_flex/rec_ot` (how we fill it), `rec_cost`, `rec_p_meet` (chance of meeting target), `rec_expected_metric` (expected minutes), `rec_risk` (Low/Medium/High), `change_vs_draft`, `why` (plain-English reason).

## Deploying (free public link)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → this repo → branch `main` → main file `app.py` → **Deploy**.
3. Every push to `main` redeploys automatically.

## Files

```
app.py                     Streamlit tool
engine/                    forecast, service model, optimizer, costing, submission
config/parameters.yaml     every assumption in one place
data/raw/                  case CSVs (unchanged)
outputs/                   plan, summary, frontier, Core3Inc_submission.csv
scripts/                   run_plan.py, backtest.py
```

## Assumptions to state in the deck

- Holiday lifts come from one analog (MLK); Valentine's Day has none (default: no extra effect; adjustable in the tool).
- Callout rates are historical by department and weekday/weekend; half of callouts are backfilled (ASSUMPTIONS.md §3).
- The service model assumes the future behaves like the six-week history.
- `team_id` is "Core 3 Inc." (confirm the format with organizers).
