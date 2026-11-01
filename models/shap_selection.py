"""SHAP-based feature selection on last walk-forward fold."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from config import LGB_PARAMS, CACHE
from validation.walk_forward import split_features

df = pd.read_parquet(CACHE / "features.parquet")
folds = split_features(df)
tr, te, trd, ted = folds[-1]

exclude = {"date", "stock_id", "close", "fwd_ret_20", "industry"}
cand = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

Xtr = tr[cand]; ytr = tr["fwd_ret_20"].values
Xte = te[cand]; yte = te["fwd_ret_20"].values
model = LGBMRegressor(**LGB_PARAMS)
model.fit(Xtr, ytr)

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(Xte)
imp = np.abs(shap_vals).mean(axis=0)
ranking = pd.DataFrame({"feature": cand, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
print(ranking.to_string(index=False))
ranking.to_parquet(CACHE / "shap_importance.parquet", index=False)