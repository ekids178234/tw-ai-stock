"""Streamlit dashboard for Taiwan AI stock selection system."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Taiwan AI Stock Selection", layout="wide")
st.title("Taiwan AI Stock Selection - Dashboard")

CACHE = Path(__file__).resolve().parent / "cache"

@st.cache_data
def load(name):
    return pd.read_parquet(CACHE / f"{name}.parquet")

# --- Backtest summary ---
st.header("Backtest (OOS)")
try:
    res = load("backtest_periods")
    cum = (1 + res["net_ret"]).cumprod()
    st.line_chart(cum.rename("cumulative net return"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{cum.iloc[-1]-1:.1%}")
    c2.metric("Win rate", f"{(res['net_ret']>0).mean():.1%}")
    c3.metric("Avg period return", f"{res['net_ret'].mean():.2%}")
    c4.metric("Periods", len(res))
    st.dataframe(res.tail(10), use_container_width=True)
except Exception as e:
    st.warning(f"no backtest data: {e}")

# --- Fold metrics ---
st.header("Walk-forward fold metrics")
try:
    # recompute quick from oos predictions? show feature importance instead
    st.info("See models/train.py output for fold IC table.")
except Exception:
    pass

# --- Latest picks (historical OOS) ---
st.header("Latest top-20 picks - historical OOS")
try:
    import joblib
    model = joblib.load(CACHE / "model_final.joblib")
    cols = joblib.load(CACHE / "feature_cols.joblib")
    feats = load("features")
    last_date = feats["date"].max()
    snap = feats[feats["date"] == last_date].copy()
    X = snap[cols].values
    snap["pred"] = model.predict(X)
    picks = snap.sort_values("pred", ascending=False).head(20)[
        ["stock_id", "pred", "ret20", "mr_yoy", "roe", "net_margin"]
    ]
    st.dataframe(picks, use_container_width=True)
    st.caption(f"signal date: {last_date.date()}")
except Exception as e:
    st.warning(f"could not load model/picks: {e}")

# --- Recent picks (2024+ via FinMind) ---
st.header("Recent top-20 picks (2024+ via FinMind)")
try:
    rp = pd.read_parquet(CACHE / "recent_picks.parquet")
    st.dataframe(rp[["stock_id","pred"]].head(20), use_container_width=True)
    st.caption("Quarterly fundamentals imputed as NaN (lower confidence).")
except Exception as e:
    st.warning(f"no recent picks: {e}")

# --- Feature importance ---
st.header("Feature importance")
try:
    import joblib
    model = joblib.load(CACHE / "model_final.joblib")
    cols = joblib.load(CACHE / "feature_cols.joblib")
    imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    st.bar_chart(imp.set_index("feature"))
except Exception as e:
    st.warning(f"no importance: {e}")