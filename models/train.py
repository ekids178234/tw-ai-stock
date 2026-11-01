"""Train LightGBM with walk-forward CV; report IC and top-decile returns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor
from config import LGB_PARAMS, CACHE
from validation.walk_forward import split_features

FEATURE_COLS = [
    "vol20", "ret20", "rev_growth", "profit_growth", "mr_mom",
    "ret120", "avg_turnover_60", "mr_accel", "net_margin",
    "trust_net_20", "mr_yoy", "current_ratio", "mr_yoy_prev",
    "foreign_hold", "eps",
]


def evaluate(pred, y):
    ic, _ = spearmanr(pred, y)
    # top decile return (by pred) mean fwd_ret
    order = np.argsort(pred)[::-1]
    top_n = max(1, len(pred) // 10)
    top_ret = y[order][:top_n].mean()
    bot_ret = y[order][-top_n:].mean()
    return ic, top_ret, bot_ret


def main():
    df = pd.read_parquet(CACHE / "features.parquet")
    folds = split_features(df)
    oos_preds = []
    fold_metrics = []
    for i, (tr, te, trd, ted) in enumerate(folds):
        Xtr = tr[FEATURE_COLS].values
        ytr = tr["fwd_ret_20"].values
        Xte = te[FEATURE_COLS].values
        yte = te["fwd_ret_20"].values
        model = LGBMRegressor(**LGB_PARAMS)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        ic, top_ret, bot_ret = evaluate(pred, yte)
        fold_metrics.append({
            "fold": i,
            "test_start": str(ted.min().date()),
            "test_end": str(ted.max().date()),
            "IC": round(float(ic), 4),
            "top_decile_ret": round(float(top_ret), 4),
            "bottom_decile_ret": round(float(bot_ret), 4),
            "spread": round(float(top_ret - bot_ret), 4),
            "n_test": len(te),
        })
        oos = te[["date", "stock_id", "fwd_ret_20"]].copy()
        oos["pred"] = pred
        oos_preds.append(oos)
        print(f"fold {i}: IC={ic:.4f} top10={top_ret:.4f} bot10={bot_ret:.4f} spread={top_ret-bot_ret:.4f}")

    metrics = pd.DataFrame(fold_metrics)
    print("\n=== Fold metrics ===")
    print(metrics.to_string(index=False))
    print("\nMean IC:", round(metrics["IC"].mean(), 4))
    print("Mean spread:", round(metrics["spread"].mean(), 4))

    oos_all = pd.concat(oos_preds, ignore_index=True)
    oos_all.to_parquet(CACHE / "oos_predictions.parquet", index=False)

    # train final model on all data for inference
    final = LGBMRegressor(**LGB_PARAMS)
    final.fit(df[FEATURE_COLS].values, df["fwd_ret_20"].values)
    import joblib
    joblib.dump(final, CACHE / "model_final.joblib")
    joblib.dump(FEATURE_COLS, CACHE / "feature_cols.joblib")
    print("saved model + predictions")


if __name__ == "__main__":
    main()