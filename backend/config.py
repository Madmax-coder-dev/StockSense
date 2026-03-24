"""
config.py — Central Configuration for NiftyOracle v2
All hyperparameters, constants, and paths in one place.
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOG_DIR    = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

for _d in [MODEL_DIR, LOG_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ───────────────────────────────────────────────────────────────────────
DATA_PERIOD       = "5y"
SEQ_LEN           = 60
TRAIN_SPLIT       = 0.70
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.15
TREND_THRESHOLD   = 0.003

# ── Architecture ───────────────────────────────────────────────────────────────
LSTM_HIDDEN       = 256
LSTM_LAYERS       = 4
LSTM_DROPOUT      = 0.35
ATTN_HEADS        = 8
SENTIMENT_DIM     = 6

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS            = 150
BATCH_SIZE        = 64
LEARNING_RATE     = 3e-4
WEIGHT_DECAY      = 1e-4
GRAD_CLIP         = 1.0
PATIENCE          = 25
LABEL_SMOOTHING   = 0.1
TREND_LOSS_WEIGHT = 0.55
PRICE_LOSS_WEIGHT = 0.45
USE_FOCAL_LOSS    = True

# ── Ensemble ───────────────────────────────────────────────────────────────────
ENSEMBLE_FOLDS    = 5
ENSEMBLE_WEIGHTS  = [0.30, 0.25, 0.20, 0.15, 0.10]

# ── Sentiment ──────────────────────────────────────────────────────────────────
FINBERT_MODEL      = "ProsusAI/finbert"
FALLBACK_BERT      = "distilbert-base-uncased-finetuned-sst-2-english"
NEWS_LOOKBACK_DAYS = 7
MAX_HEADLINES      = 50
SENTIMENT_DECAY    = 0.85

# ── NIFTY 50 Tickers ───────────────────────────────────────────────────────────
NIFTY50_TICKERS = {
    "RELIANCE":   "RELIANCE.NS",   "TCS":        "TCS.NS",
    "HDFCBANK":   "HDFCBANK.NS",   "INFY":       "INFY.NS",
    "ICICIBANK":  "ICICIBANK.NS",  "HINDUNILVR": "HINDUNILVR.NS",
    "ITC":        "ITC.NS",        "SBIN":       "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS", "KOTAKBANK":  "KOTAKBANK.NS",
    "LT":         "LT.NS",         "AXISBANK":   "AXISBANK.NS",
    "ASIANPAINT": "ASIANPAINT.NS", "MARUTI":     "MARUTI.NS",
    "WIPRO":      "WIPRO.NS",      "ULTRACEMCO": "ULTRACEMCO.NS",
    "NESTLEIND":  "NESTLEIND.NS",  "SUNPHARMA":  "SUNPHARMA.NS",
    "TITAN":      "TITAN.NS",      "BAJFINANCE": "BAJFINANCE.NS",
    "ONGC":       "ONGC.NS",       "NTPC":       "NTPC.NS",
    "POWERGRID":  "POWERGRID.NS",  "M&M":        "M&M.NS",
    "TECHM":      "TECHM.NS",      "HCLTECH":    "HCLTECH.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS", "DRREDDY":    "DRREDDY.NS",
    "CIPLA":      "CIPLA.NS",      "EICHERMOT":  "EICHERMOT.NS",
    "COALINDIA":  "COALINDIA.NS",  "ADANIPORTS": "ADANIPORTS.NS",
    "TATASTEEL":  "TATASTEEL.NS",  "JSWSTEEL":   "JSWSTEEL.NS",
    "BPCL":       "BPCL.NS",       "INDUSINDBK": "INDUSINDBK.NS",
    "GRASIM":     "GRASIM.NS",     "HEROMOTOCO": "HEROMOTOCO.NS",
    "DIVISLAB":   "DIVISLAB.NS",   "BRITANNIA":  "BRITANNIA.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS", "TATACONSUM": "TATACONSUM.NS",
    "HINDALCO":   "HINDALCO.NS",   "UPL":        "UPL.NS",
    "SBILIFE":    "SBILIFE.NS",    "HDFCLIFE":   "HDFCLIFE.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS", "TATAMOTOR":  "TATAMOTORS.NS",
    "ADANIENT":   "ADANIENT.NS",   "LTIM":       "LTIM.NS",
}
