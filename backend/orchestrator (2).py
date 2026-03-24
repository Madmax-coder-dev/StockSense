"""
orchestrator.py — Full Training + Evaluation Orchestrator

Run this file to train a model for any NIFTY 50 stock:
    python orchestrator.py --company RELIANCE
    python orchestrator.py --company TCS --retrain

Pipeline:
  1. Fetch & engineer features (data_pipeline.py)
  2. Compute FinBERT sentiment series (bert_sentiment.py)
  3. Build train/val/test datasets (data_pipeline.py)
  4. Train 5-fold ensemble (trainer.py)
  5. Evaluate on held-out test set (evaluator.py)
  6. Save ensemble + scalers (trainer.py)
"""

import os, sys, json, argparse
import numpy as np
from datetime import datetime

from config import (
    NIFTY50_TICKERS, MODEL_DIR, REPORT_DIR, DATA_PERIOD,
    SEQ_LEN, ENSEMBLE_FOLDS
)
from data_pipeline import (
    fetch_and_engineer, build_datasets
)
from bert_sentiment import get_daily_sentiment_series
from trainer import train_ensemble, save_ensemble, load_ensemble
from evaluator import evaluate


def is_trained(company: str) -> bool:
    path = os.path.join(MODEL_DIR, f"{company}_ensemble.pkl")
    return os.path.exists(path)


def run(company: str, retrain: bool = False):
    company = company.upper()
    ticker  = NIFTY50_TICKERS.get(company)
    if not ticker:
        raise ValueError(f"Unknown company: {company}")

    if not retrain and is_trained(company):
        print(f"[ORCHestrator] {company} already trained. Use --retrain to force.")
        return load_ensemble(company)

    print(f"\n{'='*65}")
    print(f"  NiftyOracle v2 — Training Pipeline")
    print(f"  Company : {company}  ({ticker})")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    # ── Step 1: Feature Engineering ──────────────────────────────────────────
    print(f"\n[1/5] Fetching & engineering features ({DATA_PERIOD} of data)...")
    df = fetch_and_engineer(ticker)
    print(f"      → {len(df)} trading days  |  {df.shape[1]} raw features")

    # ── Step 2: Sentiment Series ──────────────────────────────────────────────
    print(f"\n[2/5] Computing FinBERT sentiment series...")
    sentiment_series = get_daily_sentiment_series(company, ticker, len(df))
    print(f"      → sentiment shape: {sentiment_series.shape}")

    # ── Step 3: Build Datasets ────────────────────────────────────────────────
    print(f"\n[3/5] Building datasets (walk-forward split)...")
    (ds_train, ds_val, ds_test,
     scaler_X, scaler_y, df, feature_cols, X_sc, sent_arr) = \
        build_datasets(ticker, sentiment_series)

    print(f"      → features : {len(feature_cols)}")
    print(f"      → train    : {len(ds_train)}  val: {len(ds_val)}  test: {len(ds_test)}")

    # Combine train + val for K-fold CV (test always held out)
    import torch
    from torch.utils.data import ConcatDataset
    ds_trainval = ConcatDataset([ds_train, ds_val])

    # ── Step 4: Train Ensemble ─────────────────────────────────────────────────
    print(f"\n[4/5] Training {ENSEMBLE_FOLDS}-fold ensemble...")
    models = train_ensemble(
        input_size=len(feature_cols),
        ds_full=ds_trainval,
        company=company,
        n_folds=ENSEMBLE_FOLDS,
    )

    # ── Step 5: Evaluate ──────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating on held-out test set...")
    eval_report = evaluate(models, ds_test, scaler_y, company)

    # ── Save ──────────────────────────────────────────────────────────────────
    meta = {
        "company":      company,
        "ticker":       ticker,
        "feature_cols": feature_cols,
        "scaler_X":     scaler_X,
        "scaler_y":     scaler_y,
        "input_size":   len(feature_cols),
        "trained_at":   datetime.now().isoformat(),
        "data_period":  DATA_PERIOD,
        "n_train":      len(ds_train),
        "n_val":        len(ds_val),
        "n_test":       len(ds_test),
        "eval_report":  eval_report,
    }
    path = save_ensemble(models, company, meta)

    print(f"\n{'='*65}")
    print(f"  Training complete for {company}")
    print(f"  Model  → {path}")
    print(f"  Report → {REPORT_DIR}/{company}_eval.json")
    accs = [v["classification"]["accuracy"]
            for v in eval_report.get("horizons", {}).values()]
    if accs:
        print(f"  Avg trend accuracy : {sum(accs)/len(accs):.2%}")
    print(f"{'='*65}\n")
    return models, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NiftyOracle for a NIFTY 50 stock")
    parser.add_argument("--company", type=str, required=True,
                        help="Company symbol, e.g. RELIANCE, TCS, SBIN")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if model exists")
    args = parser.parse_args()
    run(args.company, retrain=args.retrain)
