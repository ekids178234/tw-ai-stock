"""Fetch FinMind chip flow (institutional net buy, margin, foreign shareholding) for 2024+."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
import pandas as pd
from FinMind.data import DataLoader
from config import CACHE
from utils import get_token


def main():
    token = get_token()
    if not token:
        print("no token"); return
    api = DataLoader()
    api.login_by_token(api_token=token)
    uni = pd.read_parquet(CACHE / "universe.parquet")["stock_id"].tolist()

    inst_rows, margin_rows, share_rows = [], [], []
    for sid in uni:
        try:
            inst = api.taiwan_stock_institutional_investors(stock_id=sid, start_date="2024-01-01", end_date="2026-12-31")
            inst_rows.append(inst)
        except Exception as e:
            print("inst err", sid, e)
        try:
            mg = api.taiwan_stock_margin_purchase_short_sale(stock_id=sid, start_date="2024-01-01", end_date="2026-12-31")
            margin_rows.append(mg)
        except Exception as e:
            print("margin err", sid, e)
        try:
            sh = api.taiwan_stock_shareholding(stock_id=sid, start_date="2024-01-01", end_date="2026-12-31")
            share_rows.append(sh)
        except Exception as e:
            print("share err", sid, e)

    # Institutional: aggregate net buy by date x stock
    if inst_rows:
        inst_all = pd.concat(inst_rows, ignore_index=True)
        inst_all["net"] = inst_all["buy"] - inst_all["sell"]
        # trust = Investment_Trust
        trust = inst_all[inst_all["name"] == "Investment_Trust"].groupby(["date", "stock_id"])["net"].sum().unstack("stock_id")
        foreign = inst_all[inst_all["name"].isin(["Foreign_Investor", "Foreign_Dealer_Self"])].groupby(["date", "stock_id"])["net"].sum().unstack("stock_id")
        trust.index = pd.to_datetime(trust.index)
        foreign.index = pd.to_datetime(foreign.index)
        trust_20 = trust.rolling(20).sum()
        foreign_20 = foreign.rolling(20).sum()
        trust_20.to_parquet(CACHE / "finmind_trust_net20.parquet")
        foreign_20.to_parquet(CACHE / "finmind_foreign_net20.parquet")
        print("inst done", trust.shape)

    if margin_rows:
        mg_all = pd.concat(margin_rows, ignore_index=True)
        mg_all["date"] = pd.to_datetime(mg_all["date"])
        mb = mg_all.pivot(index="date", columns="stock_id", values="MarginPurchaseTodayBalance")
        mb_chg5 = mb.diff(5)
        mb_chg5.to_parquet(CACHE / "finmind_margin_chg5.parquet")
        print("margin done", mb.shape)

    if share_rows:
        sh_all = pd.concat(share_rows, ignore_index=True)
        sh_all["date"] = pd.to_datetime(sh_all["date"])
        fh = sh_all.pivot(index="date", columns="stock_id", values="ForeignInvestmentSharesRatio")
        fh.to_parquet(CACHE / "finmind_foreign_hold.parquet")
        print("shareholding done", fh.shape)


if __name__ == "__main__":
    main()