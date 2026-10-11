"""Audit: which data each baseline was fitted on, for validation and for the final test run.

Reads no test labels: fit windows are all before TEST_START, and the 'mean prediction'
on test needs only the test hours' local hour / feature values, never `y`.
"""
import pandas as pd

from data import ROOT
from modeling import baselines
from splits import TEST_END, TEST_START, VAL_START, fit_rows

pd.set_option("display.width", 220)
grid = pd.read_csv(ROOT / "data" / "kyiv_hourly.csv.gz", index_col="t", parse_dates=["t"])
FEATURE_COLS = [c for c in grid.columns if c not in ("y", "n_starts_next_1h", "eda_active_next_1h")]


def audit(eval_start, eval_end, label):
    fit = fit_rows(grid, eval_start, None)
    end = fit.index.max()
    # label-free copy of the evaluation rows (y replaced by NaN, never read)
    ev = grid.loc[(grid.index >= eval_start) & (grid.index < eval_end) & (grid.label_in_gap == 0),
                  FEATURE_COLS].assign(y=float("nan"))
    preds = baselines(fit, ev, grid.assign(y=grid.y.where(grid.index < eval_start)))
    win = {"constant (train rate)": (fit.index.min(), end),
           "persistence  P(y | y[t-1])": (fit.index.min(), end),
           "yesterday  P(y | y[t-24])": (fit.index.min(), end),
           "hour-of-day rate (all)": (fit.index.min(), end),
           "hour-of-day rate (365D)": (end - pd.Timedelta("365D") + pd.Timedelta("1h"), end),
           "hour-of-day rate (182D)": (end - pd.Timedelta("182D") + pd.Timedelta("1h"), end)}
    rows = []
    for name, p in preds.items():
        if name in win:
            a, b = win[name]
            rate = fit.y[(fit.index >= a) & (fit.index <= b)].mean()
            rows.append({"baseline": name, "fit_from": a, "fit_to (last hour)": b,
                         "fit_rate": round(rate, 4), "mean_pred": round(float(p.mean()), 4)})
        else:   # last-30d rate: no fitting, rolling count of starts in [t-30d, t)
            rows.append({"baseline": name, "fit_from": "rolling", "fit_to (last hour)": "t - 1h, per row",
                         "fit_rate": "-", "mean_pred": round(float(p.mean()), 4)})
    print(f"\n{label}: evaluation from {eval_start}")
    print(pd.DataFrame(rows).to_string(index=False))


audit(VAL_START, TEST_START, "VALIDATION run (fit on train)")
audit(TEST_START, TEST_END, "FINAL TEST run (fit on train + validation)")
