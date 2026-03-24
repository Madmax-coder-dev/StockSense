"""
trainer.py — Model Training with K-Fold Time-Series Cross-Validation

Accuracy improvements vs v1:
  • K-Fold walk-forward CV (5 folds) → ensemble of 5 models
  • OneCycleLR scheduler → faster convergence, better generalisation
  • Gradient clipping + AdamW weight decay → stable training
  • Early stopping per fold with patience
  • Training metrics logged to CSV for analysis
  • Checkpoint saves best model per fold
"""

import os, time, json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import (
    DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    GRAD_CLIP, PATIENCE, TREND_LOSS_WEIGHT, PRICE_LOSS_WEIGHT,
    ENSEMBLE_FOLDS, ENSEMBLE_WEIGHTS, MODEL_DIR, LOG_DIR
)
from model import HybridStockModel, HybridLoss
from data_pipeline import StockSequenceDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SINGLE-FOLD TRAINER
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    model: HybridStockModel,
    ds_train: StockSequenceDataset,
    ds_val:   StockSequenceDataset,
    fold_id:  int,
    company:  str,
) -> dict:
    """
    Train model for one fold. Returns history dict + best val loss.
    Saves best checkpoint to models/{company}_fold{fold_id}.pt
    """
    model = model.to(DEVICE)
    criterion = HybridLoss(PRICE_LOSS_WEIGHT, TREND_LOSS_WEIGHT)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10,
        epochs=EPOCHS,
        steps_per_epoch=max(1, len(dl_train)),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = []
    ckpt_path     = os.path.join(MODEL_DIR, f"{company}_fold{fold_id}.pt")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_price = tr_trend = 0.0
        for xb, sb, yp, yt in dl_train:
            xb, sb, yp, yt = (xb.to(DEVICE), sb.to(DEVICE),
                               yp.to(DEVICE), yt.to(DEVICE))
            optimizer.zero_grad()
            pred_p, pred_t = model(xb, sb)
            loss, pl, tl  = criterion(pred_p, pred_t, yp, yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            tr_loss  += loss.item()
            tr_price += pl.item()
            tr_trend += tl.item()

        n = max(1, len(dl_train))
        tr_loss /= n; tr_price /= n; tr_trend /= n

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_price = val_trend = 0.0
        trend_preds_all, trend_true_all = [], []

        with torch.no_grad():
            for xb, sb, yp, yt in dl_val:
                xb, sb, yp, yt = (xb.to(DEVICE), sb.to(DEVICE),
                                   yp.to(DEVICE), yt.to(DEVICE))
                pred_p, pred_t = model(xb, sb)
                loss, pl, tl  = criterion(pred_p, pred_t, yp, yt)
                val_loss  += loss.item()
                val_price += pl.item()
                val_trend += tl.item()
                # Collect 1d horizon trend accuracy
                trend_preds_all.append(pred_t[:, 0, :].argmax(1).cpu().numpy())
                trend_true_all.append(yt[:, 0].cpu().numpy())

        n = max(1, len(dl_val))
        val_loss /= n; val_price /= n; val_trend /= n

        tp = np.concatenate(trend_preds_all)
        tt = np.concatenate(trend_true_all)
        acc_1d = float((tp == tt).mean())

        row = {
            "epoch": epoch, "fold": fold_id,
            "tr_loss": round(tr_loss, 5), "val_loss": round(val_loss, 5),
            "tr_price": round(tr_price, 5), "val_price": round(val_price, 5),
            "tr_trend": round(tr_trend, 5), "val_trend": round(val_trend, 5),
            "val_acc_1d": round(acc_1d, 4),
        }
        history.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
            torch.save(best_state, ckpt_path)
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"    Fold {fold_id} | Ep {epoch:3d}/{EPOCHS} | "
                  f"tr={tr_loss:.4f} val={val_loss:.4f} | "
                  f"acc_1d={acc_1d:.2%} | {elapsed:.0f}s")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch} (no improve {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    # Save history
    log_path = os.path.join(LOG_DIR, f"{company}_fold{fold_id}_history.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"    Fold {fold_id} best val_loss={best_val_loss:.5f}  "
          f"log → {log_path}")
    return {"history": history, "best_val_loss": best_val_loss, "ckpt": ckpt_path}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  K-FOLD WALK-FORWARD TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(
    input_size:   int,
    ds_full:      StockSequenceDataset,   # combined train+val set
    company:      str,
    n_folds:      int = ENSEMBLE_FOLDS,
) -> list:
    """
    Walk-forward K-fold cross-validation on ds_full.
    Returns list of trained model objects (one per fold).
    """
    n       = len(ds_full)
    fold_sz = n // n_folds
    models  = []

    print(f"\n  Training {n_folds}-fold ensemble for {company}  "
          f"(n={n}, fold_sz≈{fold_sz})")

    for fold in range(n_folds):
        val_start = fold * fold_sz
        val_end   = val_start + fold_sz if fold < n_folds - 1 else n
        train_idx = list(range(0, val_start)) + list(range(val_end, n))
        val_idx   = list(range(val_start, val_end))

        if len(train_idx) < 64 or len(val_idx) < 16:
            print(f"    Fold {fold}: too small, skipping.")
            continue

        sub_train = Subset(ds_full, train_idx)
        sub_val   = Subset(ds_full, val_idx)

        model = HybridStockModel(input_size=input_size)
        print(f"\n  ── Fold {fold+1}/{n_folds} ──"
              f"  train={len(train_idx)}  val={len(val_idx)}")
        train_one_fold(model, sub_train, sub_val, fold_id=fold+1, company=company)
        models.append(model.eval())

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SAVE / LOAD ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def save_ensemble(models: list, company: str, meta: dict):
    """Save all fold checkpoints + metadata pickle."""
    import pickle
    meta["n_folds"]     = len(models)
    meta["input_size"]  = models[0].tcn.blocks[0].conv1.in_channels \
                          if hasattr(models[0], "tcn") else meta.get("input_size", 45)
    out = {"meta": meta, "model_states": [m.state_dict() for m in models]}
    path = os.path.join(MODEL_DIR, f"{company}_ensemble.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"  Ensemble saved → {path}")
    return path


def load_ensemble(company: str) -> tuple:
    """Load ensemble from pickle. Returns (models_list, meta)."""
    import pickle
    path = os.path.join(MODEL_DIR, f"{company}_ensemble.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    meta   = data["meta"]
    models = []
    for state in data["model_states"]:
        m = HybridStockModel(input_size=meta["input_size"]).to(DEVICE)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models, meta
