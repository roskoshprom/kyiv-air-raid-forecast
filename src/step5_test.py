"""Step 5: the ONE final evaluation on the locked test window (+ supplementary tail).

Nothing here is tuned: model config and the two reference baselines come from
results/selected_config.json written by step 4 (validation only).
Run only with KSE_UNLOCK_TEST=1 (run_all.py sets it for this script alone).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import ROOT
from modeling import baselines, block_bootstrap_diff, make_X, make_model, metrics, reliability
from splits import TEST_END, TEST_START, fit_rows, unlock_test

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
RES, FIG = ROOT / "results", ROOT / "figures"
HIST = {"all": None, "12m": "365D", "6m": "182D"}

cfg = json.loads((RES / "selected_config.json").read_text())
grid = pd.read_csv(ROOT / "data" / "kyiv_hourly.csv.gz", index_col="t", parse_dates=["t"])

# ---------------------------------------------------------------- final fit: train + validation
fit_df = fit_rows(grid, TEST_START, HIST[cfg["history"]])
model = make_model(cfg["model"])
model.fit(make_X(fit_df, cfg["features"]), fit_df.y)
base_fit = fit_rows(grid, TEST_START, None)
print(f"final model: {cfg} | fit rows {len(fit_df)}, last fit hour {fit_df.index.max()}")

coef = pd.Series(model[-1].coef_[0], index=make_X(fit_df.head(), cfg["features"]).columns)
print("\nstandardised logistic-regression coefficients:\n", coef.sort_values().round(3).to_string())
coef.to_csv(RES / "final_model_coefficients.csv", header=["coef"])


def evaluate(df, label):
    p_model = model.predict_proba(make_X(df, cfg["features"]))[:, 1]
    preds = {f"MODEL: {cfg['model']} | {cfg['history']} | {cfg['features']}": p_model,
             **baselines(base_fit, df, grid)}
    tab = pd.DataFrame({k: metrics(df.y, p) for k, p in preds.items()}).T
    print(f"\n{label}: {len(df)} h, {int(df.y.sum())} positive hours, rate {df.y.mean():.4f}")
    print(tab.drop(columns="acc@0.5").round(4).to_string())
    return tab, preds, p_model


# ---------------------------------------------------------------- test (opened once)
test = unlock_test(grid, reason="final evaluation, step5_test.py", part="test")
assert test.index.min() >= TEST_START and test.index.max() < TEST_END
assert fit_df.index.max() < test.index.min()
test_tab, test_preds, p_test = evaluate(test, "TEST [2026-08-01 04:00, 2026-08-29 04:00)")
test_tab.to_csv(RES / "test_metrics.csv", index_label="model")

print("\nPaired block bootstrap over days (model - baseline), 2000 resamples:")
boot = {}
for b in [cfg["best_baseline_logloss"], cfg["best_baseline_auc"]]:
    ci = block_bootstrap_diff(test.y, p_test, test_preds[b], test.index)
    boot[b] = ci
    print(f"  vs [{b}]: logloss diff 95% CI {np.round(ci['logloss_diff_CI'][:2], 4)}, "
          f"P(model better)={ci['logloss_diff_CI'][2]:.2f} | ROC-AUC diff 95% CI "
          f"{np.round(ci['auc_diff_CI'][:2], 3)}, P(model better)={1 - ci['auc_diff_CI'][2]:.2f}")
(RES / "test_bootstrap.json").write_text(json.dumps(boot, indent=2))

rel = reliability(test.y, p_test)
print("\nRELIABILITY on test (5 quantile bins):\n", rel.round(4).to_string())
exp_pos = p_test.sum()
print(f"expected positives {exp_pos:.1f}, observed {int(test.y.sum())}, "
      f"z = {(test.y.sum() - exp_pos) / np.sqrt(exp_pos):.2f}")

# ---------------------------------------------------------------- tail (supplementary)
tail = unlock_test(grid, reason="supplementary tail check, step5_test.py", part="tail")
print("\nTAIL CAVEAT: history features are damaged by the 29.08-02.09 data gap "
      f"(history_in_gap_7d in {int(tail.history_in_gap_7d.sum())}/{len(tail)} hours); "
      "'yesterday' baseline reads gap hours as 0 during the first 24h.")
tail_tab, _, _ = evaluate(tail, "TAIL [2026-09-02 14:00, 2026-09-07 05:00)")
tail_tab.to_csv(RES / "tail_metrics.csv", index_label="model")

# ---------------------------------------------------------------- figures
SURFACE, INK2, MUTED, GRIDC, BLUE, ORANGE = "#fcfcfb", "#52514e", "#898781", "#e1e0d9", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRIDC,
                     "axes.grid": True, "grid.color": GRIDC, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "legend.frameon": False})

fig, ax = plt.subplots(figsize=(5.2, 4.6))
b = cfg["best_baseline_logloss"]
for p, c, lab in [(p_test, BLUE, "selected model"), (test_preds[b], ORANGE, f"baseline: {b}")]:
    r = reliability(test.y, p)
    ax.plot(r.mean_pred, r.observed, color=c, lw=2, marker="o", ms=7, label=lab)
lim = 0.3
ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("mean predicted probability (quintile bins)")
ax.set_ylabel("observed share of hours with a new alert")
ax.set_title("Calibration on test")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "test_reliability.png", dpi=150)

day = test.index.tz_convert("Europe/Kyiv").floor("D")
daily = pd.DataFrame({"observed": test.y.values, "model": p_test,
                      b: test_preds[b]}, index=day).groupby(level=0).sum()
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.bar(daily.index, daily.observed, width=0.8, color=GRIDC, label="observed hours with a new alert")
ax.plot(daily.index, daily.model, color=BLUE, lw=2, marker="o", ms=4, label="selected model (expected)")
ax.plot(daily.index, daily[b], color=ORANGE, lw=2, ls="--", label=f"{b} (expected)")
ax.set_ylabel("hours per day")
ax.set_title("Test window: expected vs observed alert-start hours per day (Kyiv dates)")
ax.legend(loc="upper left", ncol=3, fontsize=8)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIG / "test_daily.png", dpi=150)
print("\nsaved results/test_metrics.csv, results/tail_metrics.csv, figures/test_*.png")
