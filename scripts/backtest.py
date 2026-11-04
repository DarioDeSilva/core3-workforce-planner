"""Forecast accuracy check: our method vs the naive same-weekday benchmark on held-out weeks.

    python scripts/backtest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.forecast import backtest, estimate_lifts  # noqa: E402
from engine import data  # noqa: E402

if __name__ == "__main__":
    print("Holiday lifts estimated from the MLK analog:")
    for k, v in estimate_lifts(data.history()).items():
        print(f"  {k}: {v}")
    bt = backtest()
    print("\nShift-level MAPE (lower is better):")
    print(bt.to_string())
    print(f"\nAverage: ours {bt.mape_ours.mean():.1%} vs naive {bt.mape_naive.mean():.1%}")
