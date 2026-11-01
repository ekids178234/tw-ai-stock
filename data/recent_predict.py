"""Build recent (2024+) features from FinMind and predict with trained model."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
import pandas as pd
import joblib
from FinMind.data import DataLoader
from config import CACHE
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import get_token

def main():
    token = get_token()
    if not token:
        print("no token"); return
    api = DataLoader()
    api.login_by_token(api_token=token)

    close = pd.read_parquet(CACHE / "finmind_recent_close.parquet")
    close.index = pd.to_datetime(close.index)
    trading_dates = close.index
    universe = [c for c in close.columns]

    # fetch monthly revenue for universe (2024+)
    mr_rows = []
    for sid in universe:
        try:
            df = api.taiwan_stock_month_revenue(stock_id=sid, start_date="2024-01-01", end_date="2026-12-31")
            mr_rows.append(df)
        except Exception as e:
            print("err", sid, e)
    mr_all = pd.concat(mr_rows, ignore_index=True)
    mr_all["date"] = pd.to_datetime(mr_all["date"])
    # pivot revenue by stock
    mr_pivot = mr_all.pivot(index="date", columns="stock_id", values="revenue")
    # YoY growth (12 months)
    mr_yoy = mr_pivot.pct_change(12)

    # build features for latest date
    last_date = trading_dates.max()
    # momentum features
    ret5 = close.pct_change(5).loc[last_date]
    ret20 = close.pct_change(20).loc[last_date]
    ret60 = close.pct_change(60).loc[last_date]
    ret1 = close.pct_change(1).loc[last_date]
    vol20 = close.pct_change(1).rolling(20).std().loc[last_date]
    vol = close  # placeholder
    vol_ma20 = close.rolling(20).std()  # not volume; use price vol as proxy
    vol_ratio = (close.pct_change().rolling(5).std() / close.pct_change().rolling(20).std()).loc[last_date]

    # monthly revenue aligned
    mr_avail = mr_pivot[mr_pivot.index <= last_date]
    mr_latest = mr_avail.iloc[-1] if len(mr_avail) else pd.Series(index=universe, dtype=float)
    mr_yoy_latest = mr_yoy[mr_yoy.index <= last_date].iloc[-1] if len(mr_yoy[mr_yoy.index <= last_date]) else pd.Series(index=universe, dtype=float)

    # FinMind quarterly fundamentals (PIT)
    try:
        fdf = pd.read_parquet(CACHE / "finmind_fundamentals.parquet")
        fdf["disclose"] = pd.to_datetime(fdf["disclose"])
        fdf = fdf[fdf["disclose"] <= last_date]
        idx = fdf.groupby("stock_id")["disclose"].idxmax()
        f_latest = fdf.loc[idx].set_index("stock_id")
    except Exception:
        f_latest = pd.DataFrame()

    # FinMind chip flow (latest available)
    def _latest_chip(path):
        try:
            df = pd.read_parquet(CACHE / path)
            df.index = pd.to_datetime(df.index)
            df = df[df.index <= last_date]
            return df.iloc[-1] if len(df) else pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)
    trust_net20_latest = _latest_chip("finmind_trust_net20.parquet")
    foreign_net20_latest = _latest_chip("finmind_foreign_net20.parquet")
    foreign_hold_latest = _latest_chip("finmind_foreign_hold.parquet")
    margin_chg5_latest = _latest_chip("finmind_margin_chg5.parquet")
    if len(margin_chg5_latest):
        margin_chg5_latest = margin_chg5_latest.diff(5) if False else margin_chg5_latest

    cols = joblib.load(CACHE / "feature_cols.joblib")
    model = joblib.load(CACHE / "model_final.joblib")

    rows = []
    for s in universe:
        r = {
            "stock_id": s,
            "ret1": ret1.get(s, np.nan),
            "ret5": ret5.get(s, np.nan),
            "ret20": ret20.get(s, np.nan),
            "ret60": ret60.get(s, np.nan),
            "ret120": np.nan,
            "vol20": vol20.get(s, np.nan),
            "vol_ratio": vol_ratio.get(s, np.nan),
            "mr": mr_latest.get(s, np.nan),
            "mr_yoy": mr_yoy_latest.get(s, np.nan),
            "mr_mom": np.nan,
            "mr_accel": np.nan,
            "mr_yoy_prev": np.nan,
            "roe": float(f_latest.loc[s, "roe"]) if (len(f_latest) and s in f_latest.index) else np.nan,
            "net_margin": float(f_latest.loc[s, "net_margin"]) if (len(f_latest) and s in f_latest.index) else np.nan,
            "op_margin": float(f_latest.loc[s, "op_margin"]) if (len(f_latest) and s in f_latest.index) else np.nan,
            "eps": float(f_latest.loc[s, "eps"]) if (len(f_latest) and s in f_latest.index) else np.nan,
            "rev_growth": np.nan, "profit_growth": np.nan,
            "debt_ratio": np.nan, "current_ratio": np.nan,
            "avg_turnover_60": np.nan,
            "roa": np.nan,
            "trust_net_20": float(trust_net20_latest.get(s, np.nan)) if len(trust_net20_latest) else np.nan,
            "foreign_hold": float(foreign_hold_latest.get(s, np.nan)) if len(foreign_hold_latest) else np.nan,
        }
        # within-industry rank placeholders (NaN -> model handles)
        for c in ["ret5_indrank","ret20_indrank","ret60_indrank","mr_yoy_indrank","roe_indrank","net_margin_indrank","vol20_indrank"]:
            r[c] = np.nan
        rows.append(r)
    snap = pd.DataFrame(rows)
    X = snap[cols].values
    snap["pred"] = model.predict(X)
    picks = snap.sort_values("pred", ascending=False).head(20)
    out = CACHE / "recent_picks.parquet"
    picks.to_parquet(out, index=False)
    print("latest signal date:", last_date.date())
    print(picks[["stock_id","pred"]].to_string(index=False))


if __name__ == "__main__":
    main()