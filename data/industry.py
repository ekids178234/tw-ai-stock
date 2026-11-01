"""Fetch industry mapping from TWSE OpenAPI and cache."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from config import CACHE


def fetch_industry_map():
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    d = r.json()
    rows = []
    for x in d:
        rows.append({"stock_id": x["公司代號"], "industry": x["產業別"]})
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / "industry_map.parquet", index=False)
    print("industry map:", df.shape, df["industry"].nunique(), "industries")
    return df


def load_industry_map():
    p = CACHE / "industry_map.parquet"
    if not p.exists():
        return fetch_industry_map()
    return pd.read_parquet(p)


if __name__ == "__main__":
    fetch_industry_map()