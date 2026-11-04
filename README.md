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
