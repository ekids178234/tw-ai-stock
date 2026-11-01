"""Configuration for Taiwan AI stock selection pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

# Data / universe
START_DATE = "2015-01-01"
END_DATE = "2023-12-29"          # finlab free tier last trading day
UNIVERSE_SIZE = 120              # top N by avg market value (excludes ETFs)
MIN_HISTORY_DAYS = 500           # min trading days to be included

# Labels
HORIZON = 20                     # forward return horizon (trading days)

# Walk-forward
TRAIN_YEARS = 3
TEST_MONTHS = 6
PURGE_DAYS = 20                  # gap between train end and test start
EMBARGO_DAYS = 20                 # gap after test (for label overlap)

# Model
LGB_PARAMS = dict(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)

# Backtest
TOP_K = 20
REBALANCE_EVERY = 20             # trading days
COST_BUY = 0.001425             # commission
COST_SELL = 0.001425 + 0.003    # commission + transaction tax
SLIPPAGE = 0.001                 # per side
RISK_FREE = 0.015                # annual, for Sharpe