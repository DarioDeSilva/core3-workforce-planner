# CLAUDE.md — Core 3 Inc. Workforce Planner

Context for Claude Code sessions working in this repo. Read this before changing anything.

## The competition
- **Event:** Texas MISA Fall 2026 Case Competition, Costco Workforce Planning case (sponsored by CGI). **All data is simulated, not real Costco data.**
- **Team:** Core 3 Inc. — Dario (tool/optimizer, repo owner), Leo (forecast + analysis, can run Python), Mia (slides + Tableau visuals, behavioral data science; her Tableau Public dashboard is a resume piece).
- **Deadlines:** slides + staffing CSV due **Friday Nov 6, 2026 night**; presentation **Saturday Nov 7**. 12–15 min talk incl. a short demo of this tool (live, with a recorded backup).
- **The ask (case PDF p.5):** how should each warehouse allocate labor across departments and shifts for the planning week **Thu Feb 12 – Wed Feb 18, 2026** to meet demand at the lowest practical cost while maintaining service — under three required scenarios: BASE, PEAK, DISRUPT.
- **Judging order (README in data pack):** service first, then labor cost, then feasibility. Infeasible = a block above capacity (regular + usable cross-trained + overtime) or below `min_staff`.
- **Deliverables:** presentation; filled `TEAM_SUBMISSION_TEMPLATE.csv` (567 rows, totals only, 0 for closed blocks); a proof of concept that takes a scenario as input and shows staffing, cost and service risk; appendix on forecast method, assumptions, cost/service vs the draft.
- The tool must go beyond a dashboard (PDF p.6): forecast demand, convert to staffing, recommend, balance cost vs service, respond to scenarios, **explain why**.

## Domain facts (from the data pack)
- 3 warehouses: 1 (Medium), 2 (Large), 4 (Large). 3 departments: 38 Front End (checkout, target wait ≤5 min), 92 Merchandising/Stocking (replenishment ≤60 min), 95 Pharmacy (Rx wait ≤20 min). 3 shifts: OPEN, MID, EVE. Block = warehouse × dept × date × shift.
- Labor standards (hours per demand unit at 100%): FE 1.5, Merch 1.2, Pharmacy 0.165. Units are NOT comparable across departments.
- Cross-training only between Front End ↔ Merchandising; using one removes one from the partner's spare regular staff in that shift. **Pharmacy has none in either direction (licensure).**
- Official cost rule: regular staff first → partner's unused cross-trained staff → overtime. Loaded regular = base × 1.25; overtime = base × 1.5 × 1.10 (~32% more, not 50%).
- Scenarios: PEAK multiplies demand ×1.05–1.30 on Feb 13–15 (strongest at W4); DISRUPT = BASE demand but W2 illness Feb 13–16 (FE −30%, one pharmacy member out per block, OT halved) and W4 opening stocking crew −35% Feb 12–13. The managers' draft has 12 blocks over DISRUPT capacity.
- Holidays: Valentine's Day Sat Feb 14 (no historical analog), Presidents' Day Mon Feb 16 (warehouse open), Feb 17 post-holiday rebound.
- Ignore: `API.Employee`, model run 2026010101 (leaks actuals), daily labor bands.

## Key findings (verified from history)
- **The 0.85 cliff:** utilization = demand × standard ÷ worked hours. Blocks at 0.80–0.85 meet target ~99–100%; 0.85–0.90 ~60%; above 0.90 ~0%. Same in all departments.
- **MLK analog lifts:** pre-holiday weekend FE ×1.185, Merch ×1.15, Pharmacy ×1.00; holiday FE ×1.11, Merch ×1.12, Pharmacy ×0.74; rebound day ~×1.06.
- Callout rates ~6.1% FE, 4.1% Merch, 4.2% Pharmacy; about half backfilled.
- Chronic understaffing: W1 Merch (60% meet target), W2 FE (61%), weekend MID/EVE FE+Merch (44–56%). Overstaffed: W2 Pharmacy (98%, ~0.64 util).

## Decisions made (don't silently change these)
- Planning method: **optimizer** is the recommendation; rule-of-thumb mode exists for comparison.
- **Service goal 0.99** (`portfolio_service_target`), utilization ceiling 0.90, callout buffer included and stated as an assumption, backfill 0.5, Valentine's extra = 1.00 (adjustable judgment call).
- Data: CSVs only (no SQL Server). Python + Streamlit, hosted on Streamlit Community Cloud from `main` (core3-workforce-planner.streamlit.app). Tableau for deck visuals, built from `outputs/`.
- `team_id` = "Core 3 Inc."; submission file `outputs/Core3Inc_submission.csv` (confirm format with organizers).
- Story: the draft looks cheap only because it understaffs the holiday weekend. Ours costs more but fixes service.

## Current results (optimizer, goal 0.99)
| Scenario | Our cost | Draft cost | Shifts meeting target (ours / draft) | High-risk (ours / draft) |
|---|---|---|---|---|
| BASE | $374,788 | $329,108 | 99% / 62% | 0 / 72 |
| PEAK | $393,382 | $329,108 | 90% / 55% | 19 / 81 |
| DISRUPT | $371,428 | $326,367 | 93% / 61% | 12 / 73 |
- BASE cost-vs-service curve: 80% ≈ $336K, 98% ≈ $364K, 99% ≈ $375K. Remaining PEAK/DISRUPT risk is in blocks already at capacity.
- Forecast backtest (held-out weeks 5–6): 4.6% shift-level MAPE vs 7.7% naive.
- All 567 submission rows feasible; costs cross-checked independently.

## Code map
```
app.py                    Streamlit UI (tabs: Staffing plan, Plan vs draft, Service risk & actions, Explain a shift, Cost vs service, Method)
engine/data.py            loads CSVs + config; constants (dept ids, partner pairs)
engine/forecast.py        normal-day weekday baseline × MLK-analog holiday lifts × shift shares × scenario multiplier; backtest()
engine/service.py         ServiceModel: logistic P(meet target | utilization, warehouse) per dept + expected-minutes curve; callout_rates()
engine/staffing_core.py   expected_service(): P(meet) averaged over binomial callouts with backfill
engine/optimizer.py       greedy optimizer: Phase 0 staff every block to util ≤ ceiling (own staff first, then borrow); Phase 2 add the move with best service gain per $ until goal; take-back moves for FE/Merch pairs; frontier()
engine/costing.py         official cost rule, partner spare, capacity, feasibility
engine/staffing.py        prepare(), build_plan(mode), score_service(), explain() plain-English "why", summarize()
engine/submission.py      fill + validate the 567-row template
config/parameters.yaml    every assumption (Leo edits this)
scripts/run_plan.py       regenerate outputs (`--mode rule|optimizer`, `--frontier`)
scripts/backtest.py       forecast accuracy + lifts
outputs/                  plan_by_block.csv (Tableau source), scenario_summary.csv, cost_service_frontier.csv, Core3Inc_submission.csv
data/raw/                 case CSVs, unchanged
```

## Commands
```bash
pip install -r requirements.txt
python scripts/run_plan.py --frontier   # regenerate all outputs (~15 s)
python scripts/backtest.py
streamlit run app.py
```

## Working rules
- **Single source of truth:** every number in the deck, tool and submission comes from the engine via `outputs/`. After any engine/config change, rerun `scripts/run_plan.py --frontier` and commit the outputs.
- Always run the submission check (prints `Submission check: OK`); never ship infeasible blocks.
- `main` is what the live app runs; work on branches and open PRs. Don't push straight to `main` during the final 24 hours without Dario's OK.
- Streamlit: use `width="stretch"` (not deprecated `use_container_width`); escape `$` in markdown text (`money_md`).
- Keep explanations plain-English — judges and warehouse managers are the audience.

## Backlog (priority order)
1. **"What changed?" action list** between scenarios (e.g., "W4 FE Fri MID: +6 (4 OT, 2 cross-trained), +$1,050") — answers PDF p.4–5 "what actions when conditions change."
2. **Mid-week re-plan panel:** enter actual traffic deviation / callouts → re-optimize remaining shifts and show changes. Best live-demo moment.
3. **Faster first load:** precompute default plans so a cold app opens fast.
4. Whole-warehouse GM view (all 3 depts × 7 days, colored by risk).
5. Plain-language labels/tooltips; fix hard-to-read yellow heatmap cell in dark mode and the confusing "draft: 0" arrow.
6. Forecast-accuracy chart in Method tab; holiday-lift slider; per-department service goals; budget mode; wait-time objective option (only if organizers score avg wait); printable schedule.

## Open questions for organizers (sent/pending)
Deadline + submission method; is service scored on submitted staff or after simulated callouts (affects the callout buffer); scored as % shifts meeting target vs average minutes (affects optimizer objective); `team_id`/file naming; live vs recorded demo; Q&A length; deck format; Valentine's effect intentionally judgment-based?
