"""Purged walk-forward splitter for time-series CV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import TRAIN_YEARS, TEST_MONTHS, PURGE_DAYS, EMBARGO_DAYS


def make_folds(dates):
    """Given sorted unique trading dates, yield (train_idx, test_idx) as date arrays."""
    dates = pd.DatetimeIndex(sorted(set(dates)))
    folds = []
    # expanding anchor: train starts at first date; test windows of TEST_MONTHS
    start = dates[0]
    # first test window after TRAIN_YEARS
    train_end_target = start + pd.DateOffset(years=TRAIN_YEARS)
    # find index of first date >= train_end_target
    i = dates.searchsorted(train_end_target)
    while True:
        train_end = dates[i - 1] if i > 0 else None
        if train_end is None:
            break
        # purge: train must end at least PURGE_DAYS before test start
        test_start_target = train_end + pd.Timedelta(days=PURGE_DAYS)
        ts = dates.searchsorted(test_start_target)
        if ts >= len(dates):
            break
        test_end_target = dates[ts] + pd.DateOffset(months=TEST_MONTHS)
        te = dates.searchsorted(test_end_target, side="right")
        if te <= ts:
            break
        # embargo: ensure next fold's train starts after test end + embargo
        train_dates = dates[:i]  # expanding: all data up to train_end
        # apply purge: drop last PURGE_DAYS worth of train? We already gap via ts.
        test_dates = dates[ts:te]
        folds.append((train_dates, test_dates))
        # advance: next train end = test end + embargo
        next_i_target = dates[te - 1] + pd.Timedelta(days=EMBARGO_DAYS) + pd.DateOffset(years=0)
        # expanding window: train grows; set i to te (test end) for next, but require purge gap handled next iter
        i = te
        if i + 60 >= len(dates):
            break
    return folds


def split_features(df):
    folds = make_folds(df["date"].unique())
    out = []
    for tr_dates, te_dates in folds:
        tr = df[df["date"].isin(tr_dates)]
        te = df[df["date"].isin(te_dates)]
        out.append((tr, te, tr_dates, te_dates))
    print(f"num folds: {len(out)}")
    for tr, te, trd, ted in out:
        print(f"  train {trd.min().date()}..{trd.max().date()} ({len(tr)})  test {ted.min().date()}..{ted.max().date()} ({len(te)})")
    return out


if __name__ == "__main__":
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / "cache" / "features.parquet")
    split_features(df)