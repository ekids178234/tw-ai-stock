"""Fetch FinMind quarterly financials (2024+) with fixed-lag PIT disclosure dates."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
import pandas as pd
from FinMind.data import DataLoader
from config import CACHE
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import get_token

# fixed lag (conservative): quarter-end -> disclosure date
LAG = {"Q1": (5, 15), "Q2": (8, 14), "Q3": (11, 14), "Q4": (3, 31)}  # month, day of next year for Q4

def disc_date(qend):
    y = qend.year
    q = (qend.month - 1) // 3 + 1
    if q == 1:
        return pd.Timestamp(year=y, month=5, day=15) + pd.Timedelta(days=1)
    if q == 2:
        return pd.Timestamp(year=y, month=8, day=14) + pd.Timedelta(days=1)
    if q == 3:
        return pd.Timestamp(year=y, month=11, day=14) + pd.Timedelta(days=1)
    # Q4 annual -> next year Mar 31
    return pd.Timestamp(year=y + 1, month=3, day=31) + pd.Timedelta(days=1)


def main():
    token = get_token()
    if not token:
        print("no token"); return
    api = DataLoader()
    api.login_by_token(api_token=token)
    uni = pd.read_parquet(CACHE / "universe.parquet")["stock_id"].tolist()

    records = []
    for sid in uni:
        try:
            df = api.taiwan_stock_financial_statement(stock_id=sid, start_date="2024-01-01", end_date="2026-12-31")
        except Exception as e:
            print("err", sid, e); continue
        if df.empty:
            continue
        # pivot
        p = df.pivot_table(index="date", columns="type", values="value", aggfunc="first")
        p.index = pd.to_datetime(p.index)
        for qend, row in p.iterrows():
            rec = {"stock_id": sid, "qend": qend}
            rev = row.get("Revenue", np.nan)
            gp = row.get("GrossProfit", np.nan)
            oi = row.get("OperatingIncome", np.nan)
            ni = row.get("IncomeAfterTaxes", np.nan)
            eq = row.get("EquityAttributableToOwnersOfParent", np.nan)
            eps = row.get("EPS", np.nan)
            rec["revenue"] = rev
            rec["gross_margin"] = gp / rev * 100 if pd.notna(rev) and rev != 0 else np.nan
            rec["op_margin"] = oi / rev * 100 if pd.notna(rev) and rev != 0 else np.nan
            rec["net_margin"] = ni / rev * 100 if pd.notna(rev) and rev != 0 else np.nan
            rec["eps"] = eps
            rec["roe"] = ni / eq * 100 if pd.notna(eq) and eq != 0 else np.nan
            rec["disclose"] = disc_date(qend)
            records.append(rec)

    out = pd.DataFrame(records)
    out.to_parquet(CACHE / "finmind_fundamentals.parquet", index=False)
    print("saved", out.shape)


if __name__ == "__main__":
    main()