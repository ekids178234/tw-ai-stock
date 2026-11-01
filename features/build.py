"""Build a point-in-time-safe feature matrix from cached finlab panels."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from config import CACHE, START_DATE, END_DATE, HORIZON

# --- helpers ---

def load(name):
    return pd.read_parquet(CACHE / f"{name}.parquet")


def to_daily_ffill(monthly_df, trading_index, shift_days=1):
    """Reindex a monthly (date x stock) frame to daily with forward fill,
    shifting availability by shift_days (conservative: available next day)."""
    out = pd.DataFrame(index=trading_index, columns=monthly_df.columns, dtype=float)
    # place each monthly row at its date + shift
    m = monthly_df.copy()
    m.index = pd.to_datetime(m.index) + pd.Timedelta(days=shift_days)
    # for each column, reindex
    for c in monthly_df.columns:
        s = m[c].dropna()
        # map to trading days: asof
        idx = trading_index.searchsorted(s.index, side="right") - 1
        # build series aligned to trading_index
        vals = pd.Series(index=trading_index, dtype=float)
        # simpler: use reindex with method ffill on combined index
        combined = s.reindex(trading_index.union(s.index).sort_values())
        combined = combined.ffill()
        out[c] = combined.reindex(trading_index)
    return out


def pit_quarterly(fund_df, disc_df, trading_index, stock):
    """Return a daily Series for one stock from quarterly fundamentals,
    available only from disclosure date (+1 day)."""
    if stock not in fund_df.columns or stock not in disc_df.columns:
        return pd.Series(index=trading_index, dtype=float)
    disc_aligned = disc_df.reindex(fund_df.index)
    d = pd.DataFrame({
        "disclose": pd.to_datetime(disc_aligned[stock].values),
        "value": fund_df[stock].values,
    }, index=fund_df.index)
    d = d.dropna(subset=["disclose", "value"])
    if d.empty:
        return pd.Series(index=trading_index, dtype=float)
    d = d.sort_values("disclose").drop_duplicates("disclose", keep="last")
    avail = pd.to_datetime(d["disclose"]) + pd.Timedelta(days=1)
    s = pd.Series(d["value"].values, index=avail)
    s = s[~s.index.duplicated(keep="last")]
    # reindex to trading days with ffill
    combined_index = trading_index.union(s.index).sort_values()
    out = s.reindex(combined_index).ffill()
    return out.reindex(trading_index)


def build():
    close = load("adj_close")
    opn = load("adj_open")
    hi = load("adj_high")
    lo = load("adj_low")
    vol = load("volume")
    trading_index = close.index
    universe = load("universe")["stock_id"].tolist()
    universe = [s for s in universe if s in close.columns]

    # restrict columns
    close = close[universe]
    opn = opn[universe]
    hi = hi[universe]
    lo = lo[universe]
    vol = vol[universe]

    # --- price features ---
    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)
    ret60 = close.pct_change(60)
    vol20 = ret1.rolling(20).std()
    # volume ratio
    vol_ma20 = vol.rolling(20).mean()
    vol_ratio = vol / vol_ma20

    # --- chip flow features (finlab) ---
    try:
        from finlab import data as flab_chip
        foreign_net = flab_chip.get("institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)")
        trust_net = flab_chip.get("institutional_investors_trading_summary:投信買賣超股數")
        foreign_hold = flab_chip.get("foreign_investors_shareholding:全體外資及陸資持股比率")
        margin_bal = flab_chip.get("margin_transactions:融資今日餘額")
        # restrict to universe & dates
        foreign_net = foreign_net.loc[START_DATE:END_DATE, [c for c in universe if c in foreign_net.columns]]
        trust_net = trust_net.loc[START_DATE:END_DATE, [c for c in universe if c in trust_net.columns]]
        foreign_hold = foreign_hold.loc[START_DATE:END_DATE, [c for c in universe if c in foreign_hold.columns]]
        margin_bal = margin_bal.loc[START_DATE:END_DATE, [c for c in universe if c in margin_bal.columns]]
        # cumulative 5/20 day net buy
        foreign_net_5 = foreign_net.rolling(5).sum()
        foreign_net_20 = foreign_net.rolling(20).sum()
        trust_net_5 = trust_net.rolling(5).sum()
        trust_net_20 = trust_net.rolling(20).sum()
        margin_chg_5 = margin_bal.diff(5)
        chip_features = {
            "foreign_net_5": foreign_net_5,
            "foreign_net_20": foreign_net_20,
            "trust_net_5": trust_net_5,
            "trust_net_20": trust_net_20,
            "foreign_hold": foreign_hold,
            "margin_chg_5": margin_chg_5,
        }
    except Exception as e:
        print("chip fetch failed:", e)
        chip_features = {}

    # --- monthly revenue features ---
    mr = load("monthly_revenue")
    mr = mr[[c for c in universe if c in mr.columns]]
    # YoY / MoM growth already? compute from raw
    mr_yoy = mr.pct_change(12)
    mr_mom = mr.pct_change(1)
    # reindex to daily (available next day after publish)
    mr_daily = to_daily_ffill(mr, trading_index, shift_days=1)
    mr_yoy_daily = to_daily_ffill(mr_yoy, trading_index, shift_days=1)
    mr_mom_daily = to_daily_ffill(mr_mom, trading_index, shift_days=1)

    # --- quarterly fundamentals (PIT via disclosure dates) ---
    disc = load("disclosure_dates")
    disc = disc[[c for c in universe if c in disc.columns]]
    fund_files = {
        "roe": "fund_ROE稅後",
        "roa": "fund_ROA稅後息前",
        "op_margin": "fund_營業利益率",
        "net_margin": "fund_稅後淨利率",
        "rev_growth": "fund_營收成長率",
        "profit_growth": "fund_稅後淨利成長率",
        "debt_ratio": "fund_負債比率",
        "current_ratio": "fund_流動比率",
        "eps": "fund_每股稅後淨利",
    }
    # map cached file names (mojibake-safe): find files starting with fund_
    import os
    cached = {f.replace("fund_", "").replace(".parquet", ""): f for f in os.listdir(CACHE) if f.startswith("fund_")}
    # build a name->path dict by matching suffix
    fund_daily = {k: pd.DataFrame(index=trading_index, columns=universe, dtype=float) for k in fund_files}
    for key, fname in fund_files.items():
        # find cached file whose name contains the chinese suffix
        # fname like 'fund_ROE稅後'; cached keys are the chinese part
        target_suffix = fname.replace("fund_", "")
        matched = None
        for ck, cf in cached.items():
            if ck == target_suffix or target_suffix in ck:
                matched = cf
                break
        if matched is None:
            print("MISSING fundamental:", target_suffix)
            continue
        fdf = pd.read_parquet(CACHE / matched)
        fdf = fdf[[c for c in universe if c in fdf.columns]]
        for s in universe:
            if s not in fdf.columns:
                continue
            series = pit_quarterly(fdf, disc, trading_index, s)
            fund_daily[key][s] = series.values

    # --- liquidity: avg daily turnover over 60d ---
    turnover = close * vol
    avg_turnover_60 = turnover.rolling(60).mean()

    # --- assemble long format ---
    records = []
    for s in universe:
        df = pd.DataFrame(index=trading_index)
        df["stock_id"] = s
        df["close"] = close[s].values
        df["ret1"] = ret1[s].values
        df["ret5"] = ret5[s].values
        df["ret20"] = ret20[s].values
        df["ret60"] = ret60[s].values
        df["vol20"] = vol20[s].values
        df["vol_ratio"] = vol_ratio[s].values
        df["mr"] = mr_daily[s].values if s in mr_daily.columns else np.nan
        df["mr_yoy"] = mr_yoy_daily[s].values if s in mr_yoy_daily.columns else np.nan
        df["mr_mom"] = mr_mom_daily[s].values if s in mr_mom_daily.columns else np.nan
        for cname, cdf in chip_features.items():
            if s in cdf.columns:
                df[cname] = cdf[s].values
            else:
                df[cname] = np.nan
        for key in fund_daily:
            df[key] = fund_daily[key][s].values
        df["avg_turnover_60"] = avg_turnover_60[s].values
        # label: forward 20d return
        fwd = close[s].pct_change(HORIZON).shift(-HORIZON)
        df["fwd_ret_20"] = fwd.values
        records.append(df)

    long = pd.concat(records, axis=0)
    long = long.reset_index().rename(columns={"index": "date"})
    # industry mapping
    try:
        imap = pd.read_parquet(CACHE / "industry_map.parquet")
        long = long.merge(imap, on="stock_id", how="left")
    except Exception:
        long["industry"] = "NA"
    long["industry"] = long["industry"].fillna("NA")

    # additional price features
    ret120 = close.pct_change(120)
    ret120_long = ret120.stack().rename("ret120").reset_index()
    ret120_long.columns = ["date", "stock_id", "ret120"]
    long = long.merge(ret120_long, on=["date", "stock_id"], how="left")
    # revenue acceleration: change in YoY growth (approx via groupby shift)
    long = long.sort_values(["stock_id", "date"])
    long["mr_yoy_prev"] = long.groupby("stock_id")["mr_yoy"].shift(20)
    long["mr_accel"] = long["mr_yoy"] - long["mr_yoy_prev"]

    # within-industry rank features (percentile within industry at each date)
    for col in ["ret5", "ret20", "ret60", "mr_yoy", "roe", "net_margin", "vol20"]:
        if col in long.columns:
            long[f"{col}_indrank"] = long.groupby(["date", "industry"])[col].rank(pct=True)

    # drop rows without label or core features
    long = long.dropna(subset=["fwd_ret_20", "ret20"])
    # liquidity filter
    thr = long["avg_turnover_60"].median()
    long = long[long["avg_turnover_60"] >= thr * 0.3]
    long.to_parquet(CACHE / "features.parquet", index=False)
    print("features shape:", long.shape)
    return long


if __name__ == "__main__":
    build()
