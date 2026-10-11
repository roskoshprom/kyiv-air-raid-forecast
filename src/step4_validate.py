"""Step 4: baselines and models on VALIDATION. The test window is never read here.

Selection rule (fixed before looking at results): lowest validation log-loss among
models; log-loss is a proper scoring rule, so it rewards both ranking and calibration.
Three earlier 28-day backtest windows inside train are reported for stability only.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT
from modeling import (FEATURE_SETS, baselines, block_bootstrap_diff, fit_predict, metrics,
                      reliability)
from splits import TEST_START, VAL_START, fit_rows, validation

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 100)
RES, FIG = ROOT / "results", ROOT / "figures"
RES.mkdir(exist_ok=True)

grid = pd.read_csv(ROOT / "data" / "kyiv_hourly.csv.gz", index_col="t", parse_dates=["t"])
assert grid.index.max() >= TEST_START  # grid contains test rows, but we only use splits below
HISTORIES = {"all": None, "12m": "365D", "6m": "182D"}
KINDS = ["logreg", "hgb"]
BLOCK = pd.Timedelta("28D")
FIT_LOG = []


def fit_frame(before, history):
    """Fit rows for a window starting at `before` (asserts no overlap, see splits.fit_rows)."""
    return fit_rows(grid, before, history)


def eval_window(start, end):
    """All baselines and all model configs for one evaluation window."""
    ev_df = grid[(grid.index >= start) & (grid.index < end) & (grid.label_in_gap == 0)]
    assert end <= TEST_START, "evaluation window must not touch the test period"
    rows, preds = [], {}
    base_fit = fit_frame(start, None)
    # explicit out-of-sample checks: every fit set ends before the window starts
    for h in [None, *HISTORIES.values()]:
        f = fit_frame(start, h)
        assert f.index.max() < start and not f.index.isin(ev_df.index).any()
    FIT_LOG.append({"window_start": start, "eval_rows": len(ev_df),
                    "fit_max_t": base_fit.index.max(), "fit_rows_all": len(base_fit)})
    for name, p in baselines(base_fit, ev_df, grid).items():
        rows.append({"model": name, "kind": "baseline", "history": "-", "features": "-",
                     **metrics(ev_df.y, p)})
        preds[name] = p
    for hname, h in HISTORIES.items():
        fit_df = fit_frame(start, h)
        for fset in FEATURE_SETS:
            for kind in KINDS:
                p, _ = fit_predict(kind, fset, fit_df, ev_df)
                name = f"{kind} | {hname} | {fset}"
                rows.append({"model": name, "kind": kind, "history": hname, "features": fset,
                             **metrics(ev_df.y, p)})
                preds[name] = p
    return pd.DataFrame(rows), preds, ev_df


# ---------------------------------------------------------------- validation
val_tab, val_preds, val_df = eval_window(VAL_START, TEST_START)
assert val_df.index.equals(validation(grid).index)
print(f"VALIDATION [{VAL_START} .. {TEST_START}): {len(val_df)} h, "
      f"{int(val_df.y.sum())} positive hours, rate {val_df.y.mean():.4f}")

# ---------------------------------------------------------------- backtest (stability only)
bt = []
for k in [3, 2, 1]:
    s = VAL_START - k * BLOCK
    tab, _, d = eval_window(s, s + BLOCK)
    tab["window"] = f"{s:%Y-%m-%d}"
    bt.append(tab)
    print(f"backtest window {s:%Y-%m-%d}: {len(d)} h, {int(d.y.sum())} positive")
bt = pd.concat(bt)
print("\nOUT-OF-SAMPLE CHECK (fit data per evaluation window):")
print(pd.DataFrame(FIT_LOG).to_string(index=False))
bt_mean = bt.groupby("model")[["ROC-AUC", "LogLoss"]].mean().add_prefix("bt_")

tab = val_tab.set_index("model").join(bt_mean)
cols = ["ROC-AUC", "PR-AUC", "Brier", "LogLoss", "mean_pred", "actual", "bt_ROC-AUC", "bt_LogLoss"]
print("\nBASELINES (validation):")
print(tab[tab.kind == "baseline"][cols].round(4).to_string())
print("\nMODELS (validation), sorted by log-loss:")
print(tab[tab.kind != "baseline"].sort_values("LogLoss")[cols].round(4).to_string())

# ---------------------------------------------------------------- selection
models = tab[tab.kind != "baseline"]
best_name = models.LogLoss.idxmin()
best = models.loc[best_name]
base = tab[tab.kind == "baseline"]
best_base = base.LogLoss.idxmin()
best_base_auc = base["ROC-AUC"].idxmax()
print(f"\nSELECTED (min validation log-loss): {best_name}")
print(f"best baseline by log-loss: {best_base} | by ROC-AUC: {best_base_auc}")
for b in sorted({best_base, best_base_auc}):
    ci = block_bootstrap_diff(val_df.y, val_preds[best_name], val_preds[b], val_df.index)
    print(f"  model - [{b}]: logloss diff 95% CI {np.round(ci['logloss_diff_CI'][:2], 4)}, "
          f"P(model better)={ci['logloss_diff_CI'][2]:.2f} | AUC diff 95% CI "
          f"{np.round(ci['auc_diff_CI'][:2], 3)}")
# best per model kind, for the record
for kind in KINDS:
    print(f"best {kind}: {models[models.kind == kind].LogLoss.idxmin()}")

print("\nRELIABILITY, selected model (validation, 5 quantile bins):")
print(reliability(val_df.y, val_preds[best_name]).round(4).to_string())
print("\nRELIABILITY, best baseline:")
print(reliability(val_df.y, val_preds[best_base]).round(4).to_string())

# ---------------------------------------------------------------- save
tab.reset_index().to_csv(RES / "validation_metrics.csv", index=False)
bt.to_csv(RES / "backtest_metrics.csv", index=False)
sel = {"model": best["kind"], "history": best["history"], "features": best["features"],
       "rule": "min validation log-loss", "best_baseline_logloss": best_base,
       "best_baseline_auc": best_base_auc}
(RES / "selected_config.json").write_text(json.dumps(sel, indent=2))
print("\nsaved", sel)

# reliability figure
SURFACE, INK2, MUTED, GRIDC, BLUE, ORANGE = "#fcfcfb", "#52514e", "#898781", "#e1e0d9", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRIDC,
                     "axes.grid": True, "grid.color": GRIDC, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "legend.frameon": False})
fig, ax = plt.subplots(figsize=(5.2, 4.6))
for name, c, lab in [(best_name, BLUE, "selected model"), (best_base, ORANGE, f"baseline: {best_base}")]:
    r = reliability(val_df.y, val_preds[name])
    ax.plot(r.mean_pred, r.observed, color=c, lw=2, marker="o", ms=7, label=lab)
lim = max(0.25, ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("mean predicted probability (quintile bins)")
ax.set_ylabel("observed share of hours with a new alert")
ax.set_title("Calibration on validation")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "validation_reliability.png", dpi=150)
