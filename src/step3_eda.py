"""Step 3: EDA. Feature conclusions use TRAIN only (< VAL_START).

The validation window appears only on the dynamics chart; the test window is locked
and shown as a blank shaded band.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import LOCAL_TZ, ROOT, load_clean_events
from splits import TEST_END, TEST_START, VAL_START, development, train

pd.set_option("display.width", 200)
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# palette (validated reference palette, light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False})


def pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


grid = pd.read_csv(ROOT / "data" / "kyiv_hourly.csv.gz", index_col="t", parse_dates=["t"])
grid.index = pd.DatetimeIndex(grid.index, tz="UTC") if grid.index.tz is None else grid.index
ev, _, _ = load_clean_events()

tr = train(grid)
tr = tr[tr.warmup_30d == 0]
tr_recent = train(grid, history="365D")
print(f"train: {tr.index.min()} -> {tr.index.max()}, {len(tr)} h, y_rate={tr.y.mean():.4f}")
print(f"train last 365d: {len(tr_recent)} h, y_rate={tr_recent.y.mean():.4f}")

# ---------------------------------------------------------------- 1. hour / weekday
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# local hour of the FORECAST HOUR [t, t+1h) is hour_local of t
hr = pd.DataFrame({"all train": tr.groupby("hour_local").y.mean(),
                   "last 365d": tr_recent.groupby("hour_local").y.mean()})
dw = pd.DataFrame({"all train": tr.groupby("dow_local").y.mean(),
                   "last 365d": tr_recent.groupby("dow_local").y.mean()})
print("\nP(y=1) by Kyiv local hour:\n", hr.round(3).T.to_string())
print("\nP(y=1) by weekday:\n", dw.set_axis(DOW).round(3).T.to_string())
by_year = tr.groupby([tr.index.tz_convert(LOCAL_TZ).year, "hour_local"]).y.mean().unstack(0)
print("\ncorrelation of hourly profiles between years:\n", by_year.corr().round(2))
night = tr.hour_local.isin([0, 1, 2, 3, 4, 5])
print("share of alert starts in local 00-06:",
      {str(y): round(g.y[g.hour_local.isin(range(6))].sum() / g.y.sum(), 3)
       for y, g in tr.groupby(tr.index.tz_convert(LOCAL_TZ).year)})
# chi-square-ish strength: max/min ratio
print("hour max/min ratio (last 365d):", round(hr["last 365d"].max() / hr["last 365d"].min(), 2),
      "| weekday max/min ratio (last 365d):", round(dw["last 365d"].max() / dw["last 365d"].min(), 2))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
for col, c in [("all train", BLUE), ("last 365d", ORANGE)]:
    a1.plot(hr.index, hr[col], color=c, lw=2, marker="o", ms=4, label=col)
a1.set_xticks(range(0, 24, 2))
a1.set_xlabel("hour of day, Kyiv local time (start of the forecast hour)")
a1.set_title("Alert starts by hour of day")
a1.set_ylabel("P(new alert in the hour)")
pct(a1)
a1.legend(loc="upper center", ncol=2)
w = 0.38
x = np.arange(7)
a2.bar(x - w / 2 - 0.01, dw["all train"], w, color=BLUE, label="all train")
a2.bar(x + w / 2 + 0.01, dw["last 365d"], w, color=ORANGE, label="last 365d")
a2.set_xticks(x, DOW)
a2.set_title("By weekday")
pct(a2)
a2.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIG / "eda_hour_weekday.png", dpi=150)

# ---------------------------------------------------------------- 2. dynamics
dev = development(grid)
# 28-day blocks aligned backwards from TEST_START, so the last block == validation exactly
edges = pd.date_range(end=TEST_START, periods=((TEST_START - dev.index.min()).days // 28) + 1,
                      freq="28D")
blk = pd.cut(dev.index, edges, right=False)
dyn = dev.groupby(blk, observed=True).agg(y_rate=("y", "mean"), hours=("y", "size"),
                                          starts=("n_starts_next_1h", "sum"))
dyn["start"] = [i.left for i in dyn.index]
dyn = dyn.reset_index(drop=True)
dyn["part"] = np.where(dyn.start >= VAL_START, "validation", "train")
print("\n28-day blocks, last 8:\n", dyn.tail(8).to_string())
tr_y = train(grid).y.mean()
print(f"train mean {tr_y:.4f} | last 6 blocks of train mean "
      f"{dyn[dyn.part == 'train'].tail(6).y_rate.mean():.4f} | validation {dyn.y_rate.iloc[-1]:.4f}")

fig, ax = plt.subplots(figsize=(11, 3.8))
colors = np.where(dyn.part == "validation", ORANGE, BLUE)
ax.bar(dyn.start, dyn.y_rate, width=pd.Timedelta("26D"), align="edge", color=colors)
ax.axvspan(TEST_START, TEST_END, color=GRID, alpha=0.7, lw=0)
ax.text(TEST_START + (TEST_END - TEST_START) / 2, ax.get_ylim()[1] * 0.92, "test\n(locked)",
        ha="center", va="top", color=INK2, fontsize=9)
ax.axhline(tr_y, color=MUTED, lw=1, ls="--")
ax.text(pd.Timestamp("2023-08-20", tz="UTC"), tr_y + 0.002, f"train mean {tr_y:.1%}", va="bottom", color=INK2, fontsize=9)
ax.set_title("Share of hours with a new alert, 28-day blocks (Kyiv City)")
pct(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="train"), Patch(color=ORANGE, label="validation"),
                   Patch(color=GRID, label="test (locked)")], loc="upper left", ncol=3)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIG / "eda_dynamics.png", dpi=150)

# ---------------------------------------------------------------- 3. durations
ev_tr = ev[ev.started_at < VAL_START].copy()
ev_tr["dur_min"] = (ev_tr.finished_at - ev_tr.started_at).dt.total_seconds() / 60
q = ev_tr.dur_min.quantile([.1, .25, .5, .75, .9, .99])
print("\nduration of alerts (train), minutes:\n", q.round(1).to_dict())
print("median by year:", ev_tr.groupby(ev_tr.started_at.dt.year).dur_min.median().round(1).to_dict())
gap_next = (ev_tr.started_at.shift(-1) - ev_tr.finished_at).dt.total_seconds() / 3600
print("time from all-clear to next start, h:", gap_next.quantile([.1, .25, .5, .75]).round(2).to_dict())
print("share of next alert within 1h / 3h of all-clear:",
      round((gap_next < 1).mean(), 3), round((gap_next < 3).mean(), 3))

fig, ax = plt.subplots(figsize=(7.5, 3.6))
bins = np.logspace(np.log10(0.5), np.log10(800), 40)
ax.hist(ev_tr.dur_min, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
ax.set_xscale("log")
ax.set_xticks([1, 5, 15, 30, 60, 120, 240, 480], ["1", "5", "15", "30", "60", "120", "240", "480"])
med = q.loc[.5]
ax.axvline(med, color=INK2, lw=1, ls="--")
ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.0f} min", color=INK2, va="top", fontsize=9)
ax.set_xlabel("alert duration, minutes (log scale)")
ax.set_ylabel("alerts")
ax.set_title(f"Alert durations, Kyiv City, train ({len(ev_tr)} alerts)")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(FIG / "eda_duration.png", dpi=150)

# ---------------------------------------------------------------- 4. conditional P(y)
def rate_table(s, bins, labels):
    b = pd.cut(s, bins, labels=labels, right=False)
    return tr.groupby(b, observed=True).y.agg(["mean", "size"]).rename(
        columns={"mean": "P(y=1)", "size": "hours"})

act = tr.groupby("active_now").y.agg(["mean", "size"])
print("\nP(y=1 | alert active at t):\n", act.round(4))
el = rate_table(tr.active_elapsed_min[tr.active_now == 1].reindex(tr.index),
                [0, 30, 60, 120, 240, 1e9], ["<30m", "30–60m", "1–2h", "2–4h", ">4h"])
print("\nP(y=1 | alert active, by elapsed time):\n", el.round(4))
since_end = tr.since_last_end_h.where(tr.active_now == 0)
se = rate_table(since_end, [0, 1, 2, 3, 6, 12, 24, 48, 1e9],
                ["<1h", "1–2h", "2–3h", "3–6h", "6–12h", "12–24h", "1–2d", ">2d"])
print("\nP(y=1 | no alert, by time since last all-clear):\n", se.round(4))
lvl = tr.groupby(pd.qcut(tr.starts_30d, 5, duplicates="drop"), observed=True).y.agg(["mean", "size"])
print("\nP(y=1 | quintile of starts in last 30 days):\n", lvl.round(4))
lvl7 = tr.groupby(pd.qcut(tr.starts_7d, 5, duplicates="drop"), observed=True).y.agg(["mean", "size"])
print("\nP(y=1 | quintile of starts in last 7 days):\n", lvl7.round(4))
s24 = tr.groupby(tr.starts_24h.clip(upper=5)).y.agg(["mean", "size"])
print("\nP(y=1 | starts in last 24h, capped at 5):\n", s24.round(4))
lag24 = tr.y.shift(24)  # y one day earlier, known at t (its window ended at t-23h)
print("\nP(y=1 | y 24h ago):\n", tr.groupby(lag24).y.agg(["mean", "size"]).round(4))

fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2.2, 1.6]})
cats = ["no alert", "alert ongoing"]
axs[0].bar(cats, act["mean"].values, color=[BLUE, ORANGE], width=0.6)
axs[0].set_title("State at forecast time")
axs[1].bar(se.index.astype(str), se["P(y=1)"], color=BLUE, width=0.7)
axs[1].set_title("No alert: time since last all-clear")
axs[2].bar(el.index.astype(str), el["P(y=1)"], color=ORANGE, width=0.7)
axs[2].set_title("Alert ongoing: elapsed time")
for a in axs:
    pct(a)
    a.grid(axis="x", visible=False)
    a.axhline(tr.y.mean(), color=MUTED, lw=1, ls="--")
axs[0].set_ylabel("P(new alert in next hour)")
axs[1].text(len(se) - 0.5, tr.y.mean(), "train mean", ha="right", va="bottom", color=INK2, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "eda_conditional.png", dpi=150)
print("\nfigures saved to", FIG)
