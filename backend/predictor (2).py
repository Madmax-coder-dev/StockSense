"""
predictor.py — Inference Engine (Ensemble Prediction + Real-Time Prices)

Pipeline:
  1. Load trained ensemble from disk (or trigger training via orchestrator.py)
  2. Fetch latest real-time price + recalculate technical indicators
  3. Fetch fresh FinBERT sentiment vector
  4. Run weighted ensemble forward pass
  5. Return predictions with confidence scores for 1d / 10d / 30d
"""

import os
import pickle
import numpy as np
import torch
from datetime import datetime

from config import DEVICE, NIFTY50_TICKERS, ENSEMBLE_WEIGHTS, SEQ_LEN
from data_pipeline import fetch_and_engineer, get_realtime_price
from bert_sentiment import compute_sentiment_vector
from model import HybridStockModel
from trainer import load_ensemble

HORIZONS     = [1, 5, 10, 30]
HORIZON_IDX  = {1: 0, 5: 1, 10: 2, 30: 3}
TREND_MAP    = {0: "UP ↑", 1: "DOWN ↓", 2: "NEUTRAL →"}
TREND_KEY    = {0: "up",   1: "down",   2: "neutral"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ENSEMBLE INFERENCE ON LATEST DATA
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _ensemble_predict(
    models: list,
    X_seq:   np.ndarray,    # (60, features) — last SEQ_LEN rows, already scaled
    sent_vec: np.ndarray,   # (6,)
    scaler_y,
    weights: list = ENSEMBLE_WEIGHTS,
) -> dict:
    """Single forward pass through weighted ensemble for one input sequence."""
    w = np.array(weights[:len(models)], dtype=float)
    w /= w.sum()

    agg_price = np.zeros(len(HORIZONS), dtype=float)
    agg_probs = np.zeros((len(HORIZONS), 3), dtype=float)

    X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).to(DEVICE)      # (1, 60, F)
    s_tensor = torch.FloatTensor(sent_vec).unsqueeze(0).to(DEVICE)   # (1, 6)

    for model, wi in zip(models, w):
        model.to(DEVICE).eval()
        pred_p, pred_t = model(X_tensor, s_tensor)
        prices_sc = pred_p.squeeze(0).cpu().numpy()   # (H,)
        probs     = torch.softmax(pred_t, dim=-1).squeeze(0).cpu().numpy()  # (H, 3)
        agg_price += prices_sc * wi
        agg_probs += probs * wi

    # Inverse-scale prices
    pred_prices = np.array([
        float(scaler_y.inverse_transform([[agg_price[i]]])[0, 0])
        for i in range(len(HORIZONS))
    ])

    return {"prices": pred_prices, "probs": agg_probs}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ITERATIVE MULTI-STEP FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

def predict_horizons(
    company:       str,
    models:        list,
    scaler_X,
    scaler_y,
    feature_cols:  list,
    realtime_info: dict,
    sent_vec:      np.ndarray,
    output_horizons: list = [1, 10, 30],
) -> dict:
    """
    Fetch latest data, build rolling window, run ensemble, return predictions.
    Uses the actual real-time close price as anchor.
    """
    ticker  = NIFTY50_TICKERS[company]
    df_live = fetch_and_engineer(ticker, period="1y")  # fresh 1-year window

    X_raw    = df_live[feature_cols].values.astype(np.float32)
    X_scaled = scaler_X.transform(X_raw)

    # Always anchor to the real-time price by overwriting the Close column
    close_col = feature_cols.index("Close") if "Close" in feature_cols else 3
    rt_price  = realtime_info["price"]
    # Scale real-time price using the stored scaler
    rt_scaled = float(scaler_X.transform(
        np.array([[rt_price if i == close_col else X_scaled[-1, i]
                   for i in range(len(feature_cols))]]
                ))[0, close_col])
    X_scaled[-1, close_col] = rt_scaled

    seq = X_scaled[-SEQ_LEN:].copy()   # most recent 60 rows

    # Direct ensemble prediction (all horizons in one pass)
    res = _ensemble_predict(models, seq, sent_vec, scaler_y)
    pred_prices = res["prices"]   # indexed by HORIZONS order
    pred_probs  = res["probs"]    # (H, 3)

    current_price = rt_price
    results = {}
    for horizon in output_horizons:
        hi = HORIZON_IDX.get(horizon)
        if hi is None:
            # fallback: use nearest available horizon
            hi = min(range(len(HORIZONS)), key=lambda i: abs(HORIZONS[i] - horizon))
        price     = float(pred_prices[hi])
        probs_h   = pred_probs[hi]        # [p_up, p_down, p_neutral]
        trend_idx = int(probs_h.argmax())
        change_pct = ((price - current_price) / current_price) * 100

        results[horizon] = {
            "horizon_days":    horizon,
            "predicted_price": round(price, 2),
            "current_price":   round(current_price, 2),
            "change_pct":      round(change_pct, 2),
            "trend":           TREND_MAP[trend_idx],
            "trend_key":       TREND_KEY[trend_idx],
            "confidence": {
                "up":      round(float(probs_h[0]) * 100, 1),
                "down":    round(float(probs_h[1]) * 100, 1),
                "neutral": round(float(probs_h[2]) * 100, 1),
            },
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HIGH-LEVEL PREDICT FUNCTION (called by API)
# ─────────────────────────────────────────────────────────────────────────────

def predict(company: str) -> dict:
    """
    Entry point called by app.py.
    Loads ensemble from disk and returns full prediction payload.
    """
    company = company.upper()
    ticker  = NIFTY50_TICKERS[company]

    # Load ensemble + scalers
    models, meta = load_ensemble(company)
    scaler_X     = meta["scaler_X"]
    scaler_y     = meta["scaler_y"]
    feature_cols = meta["feature_cols"]

    # Real-time price
    print(f"  [PREDICT] Fetching real-time price for {company} ({ticker}) ...")
    rt = get_realtime_price(ticker)
    print(f"  [PREDICT] Live price: ₹{rt['price']}  ({rt['change_pct']:+.2f}%)")

    # Fresh sentiment
    print(f"  [PREDICT] Computing FinBERT sentiment ...")
    sent_vec = compute_sentiment_vector(company, ticker)

    # Predictions
    preds = predict_horizons(
        company, models, scaler_X, scaler_y,
        feature_cols, rt, sent_vec,
        output_horizons=[1, 10, 30],
    )

    # Attach evaluation report if available
    eval_report = {}
    rpath = os.path.join(
        os.path.dirname(__file__), "reports", f"{company}_eval.json"
    )
    if os.path.exists(rpath):
        import json
        with open(rpath) as f:
            eval_report = json.load(f)

    return {
        "company":      company,
        "ticker":       ticker,
        "realtime":     rt,
        "sentiment":    {
            "score":       round(float(sent_vec[0]), 4),
            "positive":    round(float(sent_vec[1]), 4),
            "negative":    round(float(sent_vec[2]), 4),
            "uncertainty": round(float(sent_vec[3]), 4),
            "news_volume": round(float(sent_vec[4]), 4),
            "momentum":    round(float(sent_vec[5]), 4),
        },
        "predictions":  preds,
        "eval_metrics": eval_report.get("horizons", {}),
        "model_meta":   {
            "n_folds":     meta.get("n_folds", 5),
            "input_size":  meta.get("input_size"),
            "trained_at":  meta.get("trained_at", "unknown"),
            "data_period": meta.get("data_period", "5y"),
        },
        "timestamp":    datetime.now().isoformat(),
    }
