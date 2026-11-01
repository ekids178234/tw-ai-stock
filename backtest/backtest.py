"""Cost-aware backtest with industry/market-cap neutral, risk-parity sizing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from config import CACHE, TOP_K, REBALANCE_EVERY, COST_BUY, COST_SELL, SLIPPAGE, RISK_FREE

def risk_parity_weights(picks, vol_series, industry_map, max_industry=0.30, cap=0.08):
    # base inverse vol
    vols = vol_series.reindex(picks).fillna(vol_series.median())
    inv = 1.0 / vols.clip(lower=1e-6)
    w = inv / inv.sum()
    # cap individual
    w = w.clip(upper=cap)
    w = w / w.sum()
    # industry cap
    ind = industry_map.reindex(picks).fillna("NA")
    for it in range(5):
        over = False
        for sector in ind.unique():
            mask = (ind == sector)
            if w[mask].sum() > max_industry:
                scale = max_industry / w[mask].sum()
                w[mask] = w[mask] * scale
                over = True
        if not over:
            break
        w = w / w.sum()
        w = w.clip(upper=cap)
        w = w / w.sum()
    return w

def main():
    oos = pd.read_parquet(CACHE / "oos_predictions.parquet")
    close = pd.read_parquet(CACHE / "adj_close.parquet")
    vol = pd.read_parquet(CACHE / "adj_close.parquet").pct_change().rolling(20).std()
    try:
        imap_df = pd.read_parquet(CACHE / "industry_map.parquet")
        industry_map = imap_df.set_index("stock_id")["industry"]
    except Exception:
        industry_map = pd.Series(dtype=str)

    trading_dates = sorted(oos["date"].unique())
    rebalance_dates = trading_dates[::REBALANCE_EVERY]

    period_returns = []
    for i, rd in enumerate(rebalance_dates):
        if i + 1 >= len(rebalance_dates):
            break
        next_rd = rebalance_dates[i + 1]
        snap = oos[oos["date"] == pd.Timestamp(rd)]
        if snap.empty:
            continue
        snap = snap.sort_values("pred", ascending=False)
        picks = snap.head(TOP_K)["stock_id"].tolist()
        # industry diversification: max 3 per industry
        if not industry_map.empty:
            ind = industry_map.reindex(picks).fillna("NA")
            keep = []
            counts = {}
            for s in picks:
                sec = ind.get(s, "NA")
                counts[sec] = counts.get(sec, 0)
                if counts[sec] < 3:
                    keep.append(s)
                    counts[sec] += 1
            picks = keep if len(keep) >= 10 else picks
        if rd not in close.index or next_rd not in close.index:
            continue
        exec_dates = close.index
        i0 = exec_dates.get_loc(rd) if rd in exec_dates else None
        if i0 is None or i0 + 1 >= len(exec_dates):
            continue
        exec0 = exec_dates[i0 + 1]
        i1 = exec_dates.get_loc(next_rd) if next_rd in exec_dates else None
        if i1 is None or i1 + 1 >= len(exec_dates):
            continue
        exec1 = exec_dates[i1 + 1]
        if exec0 not in close.index or exec1 not in close.index:
            continue
        # weights
        v_series = vol.loc[rd] if rd in vol.index else vol.iloc[-1]
        w = risk_parity_weights(picks, v_series, industry_map)
        px0 = close.loc[exec0, picks]
        px1 = close.loc[exec1, picks]
        stock_ret = (px1 / px0 - 1)
        gross = float((stock_ret * w).sum())
        cost = COST_BUY + COST_SELL + 2 * SLIPPAGE
        net = gross - cost
        period_returns.append({
            "rebalance_date": rd, "next_rebalance": next_rd,
            "gross_ret": gross, "net_ret": net, "n_picks": len(picks),
        })

    res = pd.DataFrame(period_returns)
    res.to_parquet(CACHE / "backtest_periods.parquet", index=False)
    cum = (1 + res["net_ret"]).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = (pd.Timestamp(res["next_rebalance"].iloc[-1]) - pd.Timestamp(res["rebalance_date"].iloc[0])).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else float("nan")
    periods_per_year = len(res) / years if years > 0 else 12
    volp = res["net_ret"].std() * np.sqrt(periods_per_year)
    sharpe = (ann_ret - RISK_FREE) / volp if volp > 0 else float("nan")
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    print("=== Backtest (risk-parity + industry cap) ===")
    print(f"periods: {len(res)}")
    print(f"total return: {total_ret:.2%}")
    print(f"annualized return: {ann_ret:.2%}")
    print(f"annualized vol: {volp:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"max drawdown: {mdd:.2%}")
    print(f"win rate: {(res['net_ret']>0).mean():.2%}")


if __name__ == "__main__":
    main()