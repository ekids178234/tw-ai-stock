"""End-to-end pipeline: ingest -> features -> train -> backtest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    print("=== 1. Ingest ===")
    import data.ingest as ingest
    ingest.main()
    print("=== 2. Features ===")
    import features.build as build
    build.build()
    print("=== 3. Train ===")
    import models.train as train
    train.main()
    print("=== 4. Backtest ===")
    import backtest.backtest as bt
    bt.main()
    print("=== PIPELINE DONE ===")

if __name__ == "__main__":
    main()