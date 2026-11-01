"""Optional: fetch recent (2024+) daily prices from FinMind using token in .env."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd
from FinMind.data import DataLoader
from config import CACHE
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import get_token


def main():
    token = get_token()
    if not token:
        print("no FINMIND_TOKEN; skip")
        return
    api = DataLoader()
    api.login_by_token(api_token=token)
    uni = pd.read_parquet(CACHE / "universe.parquet")["stock_id"].tolist()
    frames = []
    for sid in uni:
        try:
            df = api.taiwan_stock_daily(stock_id=sid, start_date="2024-01-01", end_date="2026-12-31")
            frames.append(df)
        except Exception as e:
            print("err", sid, e)
    if not frames:
        print("no data")
        return
    allp = pd.concat(frames, ignore_index=True)
    # pivot to date x stock for close
    close = allp.pivot(index="date", columns="stock_id", values="close")
    close.index = pd.to_datetime(close.index)
    out = CACHE / "finmind_recent_close.parquet"
    close.to_parquet(out)
    print("saved recent close:", close.shape, "->", out)


if __name__ == "__main__":
    main()