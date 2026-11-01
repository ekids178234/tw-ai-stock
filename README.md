# Taiwan AI Stock Selection System (solo build)

Leakage-aware prototype for Taiwan equities. Data: **finlab** (free, PIT disclosure dates)
+ **FinMind** (token in `.env`, recent 2024+ data).

## Pipeline
1. `data/ingest.py` - adjusted prices, monthly revenue, fundamentals, disclosure dates; top-120 universe
2. `data/industry.py` - industry map from TWSE OpenAPI
3. `features/build.py` - PIT-safe features + within-industry rank features
4. `validation/walk_forward.py` - purged walk-forward (20d purge/embargo)
5. `models/train.py` - LightGBM; OOS IC / top-decile
6. `models/feature_selection.py` - permutation importance
7. `backtest/backtest.py` - top-20, risk-parity + industry cap, cost-aware
8. `data/ingest_finmind.py` / `data/recent_predict.py` - 2024+ picks via FinMind
9. `app.py` - Streamlit dashboard

## Leakage controls
Adjusted prices; monthly revenue +1 day; quarterly fundamentals only from disclosure date +1 day;
walk-forward purge/embargo; backtest executes +1 day with full round-trip costs
(commission + 0.3% TW transaction tax).

## Results
- Baseline (27 features): Mean IC 0.0172, Sharpe 0.48, max DD -34%
- + within-industry rank features: Mean IC 0.0085 (feature selection shows roe/mr_yoy/ret60 indrank positive)
- Risk-parity + industry cap backtest: Sharpe 0.46, max DD -23.5%, win rate 54%

## Run
```
cd tw_ai_stock
python run_pipeline.py          # full pipeline (finlab data)
python data/ingest_finmind.py   # recent prices (needs token)
python data/recent_predict.py   # latest picks
streamlit run app.py            # dashboard
```
"""