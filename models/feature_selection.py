"""Permutation importance on last walk-forward fold."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
from config import LGB_PARAMS, CACHE
from validation.walk_forward import split_features

df = pd.read_parquet(CACHE / "features.parquet")
folds = split_features(df)
tr, te, trd, ted = folds[-1]

# candidate features = all numeric except ids/label
exclude = {"date", "stock_id", "close", "fwd_ret_20", "industry", "avg_turnover_60"}
cand = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
# include avg_turnover_60
cand = cand + ["avg_turnover_60"]
cand = list(dict.fromkeys(cand))

Xtr = tr[cand].values; ytr = tr["fwd_ret_20"].values
Xte = te[cand].values; yte = te["fwd_ret_20"].values
model = LGBMRegressor(**LGB_PARAMS)
model.fit(Xtr, ytr)
r = permutation_importance(model, Xte, yte, n_repeats=5, random_state=42, n_jobs=1, scoring="neg_mean_squared_error")
imp = pd.DataFrame({"feature": cand, "importance": r.importances_mean}).sort_values("importance", ascending=False)
print(imp.to_string(index=False))
imp.to_parquet(CACHE / "perm_importance.parquet", index=False)