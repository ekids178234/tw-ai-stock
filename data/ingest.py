"""Ingest Taiwan stock data from finlab into parquet cache.

finlab free tier provides clean panels (date x stock) and, crucially,
etl:financial_statements_disclosure_dates which gives the actual release date
of each quarterly report -> enables point-in-time (no-leakage) joins.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from finlab import data as flab
from config import CACHE, START_DATE, END_DATE, UNIVERSE_SIZE, MIN_HISTORY_DAYS

CACHE.mkdir(exist_ok=True)


def _save(name, df):
    p = CACHE / f"{name}.parquet"
    df.to_parquet(p)
    print(f"saved {name}: {df.shape} -> {p}")


def _load_or_fetch(name, fetcher):
    p = CACHE / f"{name}.parquet"
    if p.exists():
        print(f"cache hit: {name}")
        return pd.read_parquet(p)
    df = fetcher()
    _save(name, df)
    return df


def get_universe():
    """Top N stocks by average market value, excluding ETFs (code starts with 00)."""
    mv = _load_or_fetch("market_value", lambda: flab.get("etl:market_value"))
    # restrict to date range
    mv = mv.loc[START_DATE:END_DATE]
    # exclude ETFs
    cols = [c for c in mv.columns if not str(c).startswith("00")]
    mv = mv[cols]
    avg_mv = mv.mean(axis=0).sort_values(ascending=False)
    # require enough history
    counts = mv.notna().sum(axis=0)
    eligible = [c for c in avg_mv.index if counts.get(c, 0) >= MIN_HISTORY_DAYS]
    universe = eligible[:UNIVERSE_SIZE]
    print(f"universe size: {len(universe)}")
    _save("universe", pd.DataFrame({"stock_id": universe}))
    return universe


def ingest_prices(universe):
    close = _load_or_fetch("adj_close", lambda: flab.get("etl:adj_close"))
    o = _load_or_fetch("adj_open", lambda: flab.get("etl:adj_open"))
    hi = _load_or_fetch("adj_high", lambda: flab.get("etl:adj_high"))
    lo = _load_or_fetch("adj_low", lambda: flab.get("etl:adj_low"))
    # volume
    try:
        vol = _load_or_fetch("volume", lambda: flab.get("price:成交股數"))
    except Exception as e:
        print("volume fetch failed:", e)
        vol = None
    for name, df in [("adj_close", close), ("adj_open", o), ("adj_high", hi), ("adj_low", lo)]:
        df = df.loc[START_DATE:END_DATE, universe]
        _save(name, df)
    if vol is not None:
        vol = vol.loc[START_DATE:END_DATE, universe]
        _save("volume", vol)


def ingest_monthly_revenue(universe):
    rev = _load_or_fetch("monthly_revenue", lambda: flab.get("monthly_revenue:當月營收"))
    rev = rev.loc[:, universe] if False else rev
    # index monthly; keep columns in universe that exist
    cols = [c for c in universe if c in rev.columns]
    rev = rev[cols]
    _save("monthly_revenue", rev)


def ingest_fundamentals(universe):
    # disclosure dates (PIT release) per quarter x stock
    dd = _load_or_fetch("disclosure_dates", lambda: flab.get("etl:financial_statements_disclosure_dates"))
    cols = [c for c in universe if c in dd.columns]
    dd = dd[cols]
    _save("disclosure_dates", dd)

    # fundamental features (quarterly)
    from finlab import data as flab2
    # pick a useful subset
    wanted = [
        "fundamental_features:ROE稅後",
        "fundamental_features:ROA稅後息前",
        "fundamental_features:營業毛利率",
        "fundamental_features:營業利益率",
        "fundamental_features:稅後淨利率",
        "fundamental_features:營收成長率",
        "fundamental_features:稅後淨利成長率",
        "fundamental_features:負債比率",
        "fundamental_features:流動比率",
        "fundamental_features:每股稅後淨利",
    ]
    # verify existence
    available = set(flab2.entry_names)
    for w in wanted:
        if w not in available:
            print("MISSING dataset:", w)
    for w in wanted:
        if w not in available:
            continue
        df = _load_or_fetch(w.split(":")[1], lambda w=w: flab2.get(w))
        df = df[[c for c in cols if c in df.columns]]
        _save("fund_" + w.split(":")[1], df)


def main():
    universe = get_universe()
    ingest_prices(universe)
    ingest_monthly_revenue(universe)
    ingest_fundamentals(universe)
    print("INGEST DONE")


if __name__ == "__main__":
    main()