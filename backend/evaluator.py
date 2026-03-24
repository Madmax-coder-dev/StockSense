"""
evaluator.py — Model Evaluation on Held-Out Test Set

Computes and reports:
  • Trend accuracy (Up/Down/Neutral) per horizon
  • Weighted F1-score per horizon
  • MAE / RMSE / MAPE for price regression per horizon
  • Confusion matrix per horizon
  • Ensemble vs single-model comparison
  • Saves full report to reports/{company}_eval.json
"""

import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import DEVICE, ENSEMBLE_WEIGHTS, REPORT_DIR
from model import HybridStockModel

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        classification_report, mean_absolute_error
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


HORIZON_NAMES = {0: "1d", 1: "5d", 2: "10d", 3: "30d"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(models: list, ds_test, scaler_y, weights=None) -> dict:
    """
    Run ensemble inference on ds_test.
    Returns dict with raw arrays: pred_prices, pred_trends, true_prices, true_trends
    Shapes: (N, H) for prices/trends.
    """
    dl = DataLoader(ds_test, batch_size=128, shuffle=False)
    if weights is None:
        w = np.ones(len(models)) / len(models)
    else:
        w = np.array(weights[:len(models)], dtype=float)
        w /= w.sum()

    all_pp, all_pt = [], []   # per-model predictions
    all_tp, all_tt = [], []   # true values (collected once)
    collected = False

    for model, wi in zip(models, w):
        model.to(DEVICE).eval()
        fold_pp, fold_pt = [], []
        for xb, sb, yp, yt in dl:
            xb, sb = xb.to(DEVICE), sb.to(DEVICE)
            pp, pt = model(xb, sb)
            fold_pp.append(pp.cpu().numpy())
            fold_pt.append(torch.softmax(pt, dim=-1).cpu().numpy())
            if not collected:
                all_tp.append(yp.numpy())
                all_tt.append(yt.numpy())
        all_pp.append(np.concatenate(fold_pp, axis=0) * wi)
        all_pt.append(np.concatenate(fold_pt, axis=0) * wi)
        collected = True

    pred_prices_sc = np.sum(all_pp, axis=0)  # (N, H) weighted ensemble
    pred_probs     = np.sum(all_pt, axis=0)  # (N, H, 3)
    pred_trends    = pred_probs.argmax(axis=-1)  # (N, H)

    true_prices_sc = np.concatenate(all_tp, axis=0)  # (N, H)
    true_trends    = np.concatenate(all_tt, axis=0)  # (N, H)

    # Inverse-transform prices (only first horizon scaler is stored)
    pred_prices = scaler_y.inverse_transform(pred_prices_sc[:, :1])
    for i in range(1, pred_prices_sc.shape[1]):
        pred_prices = np.hstack([
            pred_prices,
            scaler_y.inverse_transform(pred_prices_sc[:, i:i+1])
        ])
    true_prices = scaler_y.inverse_transform(true_prices_sc[:, :1])
    for i in range(1, true_prices_sc.shape[1]):
        true_prices = np.hstack([
            true_prices,
            scaler_y.inverse_transform(true_prices_sc[:, i:i+1])
        ])

    return {
        "pred_prices": pred_prices,
        "pred_trends": pred_trends,
        "pred_probs":  pred_probs,
        "true_prices": true_prices,
        "true_trends": true_trends,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(res: dict, company: str) -> dict:
    """Compute all evaluation metrics and return as structured dict."""
    pp = res["pred_prices"]  # (N, H)
    pt = res["pred_trends"]  # (N, H)
    tp = res["true_prices"]  # (N, H)
    tt = res["true_trends"]  # (N, H)
    N, H = pp.shape

    LABEL_NAMES = ["Up", "Down", "Neutral"]
    report = {"company": company, "n_test_samples": N, "horizons": {}}

    for hi in range(H):
        hname = HORIZON_NAMES.get(hi, f"h{hi}")
        y_pred_p = pp[:, hi]
        y_pred_t = pt[:, hi]
        y_true_p = tp[:, hi]
        y_true_t = tt[:, hi]

        # Remove NaN rows
        valid = ~(np.isnan(y_true_p) | np.isnan(y_pred_p))
        y_pred_p = y_pred_p[valid]
        y_true_p = y_true_p[valid]
        y_pred_t = y_pred_t[valid]
        y_true_t = y_true_t[valid]

        if len(y_true_p) == 0:
            continue

        # Regression
        mae  = float(mean_absolute_error(y_true_p, y_pred_p)) if HAS_SKLEARN else float(np.mean(np.abs(y_true_p - y_pred_p)))
        rmse = float(np.sqrt(np.mean((y_true_p - y_pred_p) ** 2)))
        mape = float(np.mean(np.abs((y_true_p - y_pred_p) / np.clip(np.abs(y_true_p), 1e-6, None))) * 100)

        # Classification
        acc = float(accuracy_score(y_true_t, y_pred_t)) if HAS_SKLEARN else float((y_pred_t == y_true_t).mean())
        f1  = float(f1_score(y_true_t, y_pred_t, average="weighted", zero_division=0)) if HAS_SKLEARN else 0.0
        cm  = confusion_matrix(y_true_t, y_pred_t, labels=[0, 1, 2]).tolist() if HAS_SKLEARN else []

        # Per-class stats
        if HAS_SKLEARN:
            cr = classification_report(y_true_t, y_pred_t,
                                        labels=[0, 1, 2],
                                        target_names=LABEL_NAMES,
                                        output_dict=True,
                                        zero_division=0)
        else:
            cr = {}

        h_report = {
            "regression": {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape_pct": round(mape, 4)},
            "classification": {
                "accuracy": round(acc, 4),
                "f1_weighted": round(f1, 4),
                "confusion_matrix": cm,
                "per_class": cr,
            }
        }
        report["horizons"][hname] = h_report

        print(f"\n  [{hname}]  acc={acc:.2%}  f1={f1:.3f}  "
              f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
        if cm:
            print(f"         Confusion matrix (Up/Down/Neutral):")
            for row, label in zip(cm, LABEL_NAMES):
                print(f"           {label:7s}: {row}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(models: list, ds_test, scaler_y, company: str) -> dict:
    """
    Full evaluation: inference → metrics → save report.
    Returns metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {company}  (test samples = {len(ds_test)})")
    print(f"{'='*60}")

    # Ensemble inference
    res = run_inference(models, ds_test, scaler_y, weights=ENSEMBLE_WEIGHTS)

    # Metrics
    report = compute_metrics(res, company)

    # Save report
    path = os.path.join(REPORT_DIR, f"{company}_eval.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {path}")

    # Summary line
    accs = [v["classification"]["accuracy"]
            for v in report["horizons"].values()]
    print(f"\n  Summary | avg trend accuracy = {np.mean(accs):.2%}")
    return report
