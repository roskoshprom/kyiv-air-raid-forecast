"""Loading and cleaning the official sirens dataset.

All timestamps stay in UTC. Kyiv local time is used only for calendar features.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "raw" / "ukrainian-air-raid-sirens-dataset"   # see download.py
OFFICIAL_CSV = DATASET_DIR / "datasets" / "official_data_en.csv"
VOLUNTEER_CSV = DATASET_DIR / "datasets" / "volunteer_data_en.csv"

REGION = "Kyiv City"
LOCAL_TZ = "Europe/Kyiv"

# Grid end (exclusive): the official file ends 2026-09-07 05:39 UTC; we stop at 05:00
# so that every label window [t, t+1h) is fully covered by the data.
GRID_END = pd.Timestamp("2026-09-07 05:00", tz="UTC")
MERGE_GAP = pd.Timedelta("1min")      # alerts separated by < 1 min are one alert
DATA_GAP_MIN = pd.Timedelta("24h")    # country-wide silence this long = collection gap


def load_official(path=OFFICIAL_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["started_at", "finished_at"]:
        df[c] = pd.to_datetime(df[c], utc=True).dt.as_unit("ns")   # pandas 3 defaults to us
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates (a whole block of the file is repeated, see step 1),
    then assert there are no remaining duplicates by (oblast, raion, hromada, started_at)."""
    out = df.drop_duplicates().reset_index(drop=True)
    key = ["oblast", "raion", "hromada", "started_at"]
    n_key_dups = out.duplicated(subset=key).sum()   # NaN == NaN for duplicated()
    if n_key_dups:
        raise ValueError(f"{n_key_dups} rows share {key} but differ elsewhere")
    return out


def find_data_gaps(df: pd.DataFrame, min_len=DATA_GAP_MIN) -> pd.DataFrame:
    """Periods with no alert starting anywhere in the country for >= min_len.

    Assumption: in a full-scale war a country-wide silence of >= 24h is a collection
    gap, not real calm. Verified for 2026-08-29..09-02 against the volunteer file.
    """
    s = df.started_at.sort_values().reset_index(drop=True)
    gaps = pd.DataFrame({"gap_start": s.shift(), "gap_end": s})
    gaps = gaps[(gaps.gap_end - gaps.gap_start) >= min_len]
    return gaps.reset_index(drop=True)


def load_volunteer(path=VOLUNTEER_CSV) -> pd.DataFrame:
    v = pd.read_csv(path)
    for c in ["started_at", "finished_at"]:
        v[c] = pd.to_datetime(v[c], utc=True).dt.as_unit("ns")
    return v


def check_gaps_against_volunteer(official: pd.DataFrame, gaps: pd.DataFrame,
                                 start="2022-03-16", end=GRID_END) -> pd.DataFrame:
    """Justify the '>= 24h country-wide silence = collection gap' rule with data.

    Compares, per UTC day, the number of regions with >= 1 alert start in the official
    file vs the independent volunteer file (region counts are comparable across sources;
    raw row counts are not, because official switched to raion level in late 2025).
    A day is 'missing' if official has 0 alerts while volunteer has alerts in >= 5 regions.
    Asserts: every missing day lies inside a detected gap, and every full day inside a
    detected gap is missing. Returns the daily comparison table.
    """
    v = load_volunteer()
    day_o, day_v = official.started_at.dt.floor("D"), v.started_at.dt.floor("D")
    d = pd.DataFrame({"off_alerts": official.groupby(day_o).size(),
                      "off_regions": official.groupby(day_o).oblast.nunique(),
                      "vol_alerts": v.groupby(day_v).size(),
                      "vol_regions": v.groupby(day_v).region.nunique()})
    days = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end).floor("D"),
                         freq="D", inclusive="left")
    d = d.reindex(days).fillna(0).astype(int)
    d["missing"] = (d.off_alerts == 0) & (d.vol_regions >= 5)
    in_gap = np.zeros(len(d), bool)
    for g0, g1 in zip(gaps.gap_start, gaps.gap_end):
        in_gap |= (d.index >= g0.ceil("D")) & (d.index + pd.Timedelta("1D") <= g1)
    d["full_day_in_gap"] = in_gap
    bad = d[d.missing != d.full_day_in_gap]
    if len(bad):
        raise AssertionError(f"gap rule disagrees with volunteer data:\n{bad}")
    return d


def region_events(df: pd.DataFrame, region=REGION) -> pd.DataFrame:
    """Alerts of one region, sorted, with alerts separated by < MERGE_GAP merged."""
    ev = (df.loc[df.oblast == region, ["started_at", "finished_at"]]
            .sort_values("started_at").reset_index(drop=True))
    if (ev.finished_at <= ev.started_at).any():
        raise ValueError("non-positive durations in region events")
    # New group starts when the gap to the running max of previous ends is >= MERGE_GAP.
    prev_end = ev.finished_at.cummax().shift()
    new_group = ~((ev.started_at - prev_end) < MERGE_GAP)   # NaT on first row -> True
    gid = new_group.cumsum()
    merged = ev.groupby(gid).agg(started_at=("started_at", "first"),
                                 finished_at=("finished_at", "max"),
                                 n_merged=("started_at", "size"))
    merged = merged.reset_index(drop=True)
    overlaps = (merged.started_at.iloc[1:].values < merged.finished_at.iloc[:-1].values).sum()
    if overlaps:
        raise ValueError(f"{overlaps} overlapping alerts after merge")
    return merged


def load_clean_events(region=REGION, check_gaps=False):
    info = {}
    raw = load_official()
    dedup = deduplicate(raw)
    ev = region_events(dedup, region)
    gaps = find_data_gaps(dedup)
    if check_gaps:
        cmp = check_gaps_against_volunteer(dedup, gaps)
        ratio = (cmp.off_regions / cmp.vol_regions.clip(lower=1))[~cmp.full_day_in_gap]
        info["volunteer_check"] = {"days": len(cmp), "missing_days": int(cmp.missing.sum()),
                                   "min_region_ratio_outside_gaps": round(float(ratio.min()), 3)}
    info.update({"raw_rows": len(raw), "dedup_rows": len(dedup),
            "region_rows_dedup": int((dedup.oblast == region).sum()),
            "region_events_merged": len(ev),
            "merges": int((ev.n_merged > 1).sum())})
    return ev, gaps, info
