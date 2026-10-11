"""Time-based split with a locked test set.

    train       : [grid start, VAL_START)
    validation  : [VAL_START, TEST_START)        28 days — ALL decisions are made here
    test        : [TEST_START, TEST_END)         28 days — opened ONCE, at the very end
    tail        : [TAIL_START, GRID_END)         supplementary check after the data gap,
                                                 history features partly damaged by the gap

Test and tail can only be read through `unlock_test()`, which requires the environment
variable KSE_UNLOCK_TEST=1 and appends every access to results/test_access.log.
Hours whose label window touches a data gap (label_in_gap == 1) are dropped everywhere.
"""
import os
from datetime import datetime, timezone

import pandas as pd

from data import GRID_END, ROOT

VAL_START = pd.Timestamp("2026-07-04 04:00", tz="UTC")
TEST_START = pd.Timestamp("2026-08-01 04:00", tz="UTC")
TEST_END = pd.Timestamp("2026-08-29 04:00", tz="UTC")
TAIL_START = pd.Timestamp("2026-09-02 14:00", tz="UTC")   # first full hour after the gap
ACCESS_LOG = ROOT / "results" / "test_access.log"


class TestLockedError(RuntimeError):
    pass


def _clean(df):
    return df[df.label_in_gap == 0]


def development(grid: pd.DataFrame) -> pd.DataFrame:
    """Everything before the test window (train + validation). Safe for EDA."""
    return _clean(grid[grid.index < TEST_START])


def train(grid, history=None):
    """Train part. `history` (e.g. '365D') keeps only the last N days before VAL_START."""
    df = _clean(grid[grid.index < VAL_START])
    if history is not None:
        df = df[df.index >= VAL_START - pd.Timedelta(history)]
    return df


def validation(grid):
    return _clean(grid[(grid.index >= VAL_START) & (grid.index < TEST_START)])


def train_plus_validation(grid, history=None):
    """Final fit set for the test: same history length, window shifted to TEST_START."""
    df = _clean(grid[grid.index < TEST_START])
    if history is not None:
        df = df[df.index >= TEST_START - pd.Timedelta(history)]
    return df


def fit_rows(grid, before, history=None):
    """Rows usable for fitting a model that is evaluated from `before` on.

    Strictly earlier rows only, with label windows [t, t+1h) that END by `before`
    (so no label overlaps the evaluation window); gap hours and the 30-day warm-up
    are dropped; `history` optionally keeps only the last N days.
    """
    df = grid[(grid.index < before) & (grid.label_in_gap == 0) & (grid.warmup_30d == 0)]
    if history is not None:
        df = df[df.index >= before - pd.Timedelta(history)]
    assert df.index.max() + pd.Timedelta("1h") <= before, "fit rows overlap the evaluation window"
    return df


def unlock_test(grid, reason: str, part="test"):
    if os.environ.get("KSE_UNLOCK_TEST") != "1":
        raise TestLockedError("Test window is locked. Set KSE_UNLOCK_TEST=1 only for the final run.")
    if not reason:
        raise ValueError("give a reason; it is logged")
    ACCESS_LOG.parent.mkdir(exist_ok=True)
    with open(ACCESS_LOG, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{part}\t{reason}\n")
    if part == "test":
        df = _clean(grid[(grid.index >= TEST_START) & (grid.index < TEST_END)])
        # Test B must be free of gap-damaged history (checked, not assumed)
        assert df.history_in_gap_30d.sum() == 0, "test history touches a data gap"
        return df
    if part == "tail":
        return _clean(grid[(grid.index >= TAIL_START) & (grid.index < GRID_END)])
    raise ValueError(part)


def check_windows(grid):
    """Sanity checks on the split, without reading test labels."""
    assert VAL_START + pd.Timedelta("28D") == TEST_START
    assert TEST_START + pd.Timedelta("28D") == TEST_END
    test_idx = grid.index[(grid.index >= TEST_START) & (grid.index < TEST_END)]
    flags = grid.loc[test_idx, ["label_in_gap", "history_in_gap_24h",
                                "history_in_gap_7d", "history_in_gap_30d"]].sum()
    assert len(test_idx) == 672 and (flags == 0).all(), flags
    val = grid[(grid.index >= VAL_START) & (grid.index < TEST_START)]
    assert len(val) == 672 and val[["label_in_gap", "history_in_gap_30d"]].sum().sum() == 0
    return {"test_hours": len(test_idx), "test_gap_flags": flags.to_dict(),
            "val_hours": len(val)}
