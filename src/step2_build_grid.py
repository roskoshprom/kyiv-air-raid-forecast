"""Step 2: clean events and build the hourly Kyiv City grid (saved to data/)."""
import pandas as pd

from data import GRID_END, ROOT, load_clean_events
from features import FEATURES, build_grid

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

GRID_START = pd.Timestamp("2022-03-15 17:00", tz="UTC")   # first full hour of country data

ev, gaps, info = load_clean_events(check_gaps=True)
print("CLEANING:", info)
print("\nCOUNTRY-WIDE DATA GAPS (no alert starts anywhere for >= 24h):")
print(gaps.assign(hours=(gaps.gap_end - gaps.gap_start).dt.total_seconds().div(3600).round(1)))
print("\nMERGED ALERTS (gap < 1 min):")
print(ev[ev.n_merged > 1].to_string())

grid = build_grid(ev, gaps, GRID_START, GRID_END)
out = ROOT / "data" / "kyiv_hourly.csv.gz"
out.parent.mkdir(exist_ok=True)
grid.to_csv(out)

print(f"\nGRID: {len(grid)} hours, {grid.index.min()} -> {grid.index.max()} (UTC), saved {out.name}")
print("hourly continuity:", (grid.index.to_series().diff().dropna() == pd.Timedelta("1h")).all())
print("y=1 hours:", int(grid.y.sum()), "| alert starts in grid:", int(grid.n_starts_next_1h.sum()),
      "| hours with >1 start:", int((grid.n_starts_next_1h > 1).sum()))
print("flagged: label_in_gap", int(grid.label_in_gap.sum()),
      "| history_in_gap_24h", int(grid.history_in_gap_24h.sum()),
      "| history_in_gap_7d", int(grid.history_in_gap_7d.sum()),
      "| history_in_gap_30d", int(grid.history_in_gap_30d.sum()),
      "| warmup_30d", int(grid.warmup_30d.sum()))
from splits import check_windows
print("SPLIT CHECK (no test labels read):", check_windows(grid))

from splits import development, TEST_START
dev = development(grid)            # test window is locked: stats below never read it
print(f"\nSHARE y=1 BY YEAR (development part only: < {TEST_START}, gap hours excluded):")
yr = dev.groupby(dev.index.year).agg(hours=("y", "size"), y_rate=("y", "mean"),
                                     eda_active_rate=("eda_active_next_1h", "mean"))
print(yr.round(4).to_string())

# Hand-check: a multi-hour alert in the validation window
ex = ev[(ev.started_at >= "2026-07-10") & (ev.started_at < TEST_START - pd.Timedelta("1D")) &
        ((ev.finished_at - ev.started_at) > pd.Timedelta("100min"))].iloc[0]
print(f"\nHAND CHECK — alert started {ex.started_at}, finished {ex.finished_at} "
      f"(local {ex.started_at.tz_convert('Europe/Kyiv'):%H:%M}–{ex.finished_at.tz_convert('Europe/Kyiv'):%H:%M})")
near = ev[(ev.started_at > ex.started_at - pd.Timedelta("8h")) &
          (ev.started_at < ex.started_at + pd.Timedelta("8h"))]
print("neighbouring alerts:\n", near[["started_at", "finished_at"]].to_string())
t0 = ex.started_at.floor("h")
rows = grid.loc[t0 - pd.Timedelta("3h"): t0 + pd.Timedelta("6h")]
cols = ["y", "n_starts_next_1h", "eda_active_next_1h", "active_now", "active_elapsed_min",
        "since_last_start_h", "since_last_end_h", "starts_1h", "starts_24h", "starts_30d", "hour_local"]
print(rows[cols].round(2).to_string())
