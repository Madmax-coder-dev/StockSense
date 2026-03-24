"""
data_pipeline.py — Data Fetching, Feature Engineering & PyTorch Datasets

Accuracy improvements vs v1:
  • 5 years of data (vs 2y) — more patterns for the LSTM to learn
  • 45+ technical features (vs 20) — richer signal space
  • Walk-forward split — zero data leakage between train/val/test
  • RobustScaler — handles fat-tailed financial distributions
  • Gaussian noise augmentation — reduces overfitting
  • Multi-horizon targets (1d, 5d, 10d, 30d) trained simultaneously
  • Real-time price from Yahoo Finance on every predict call
"""

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

from config import (
    NIFTY50_TICKERS, DATA_PERIOD, SEQ_LEN,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, TREND_THRESHOLD
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  REAL-TIME PRICE FETCH
# ─────────────────────────────────────────────────────────────────────────────

def get_realtime_price(ticker_symbol: str) -> dict:
    """
    Fetch live/latest price data from Yahoo Finance.
    Returns current price, previous close, day change %, volume, 52w high/low.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        fast = ticker.fast_info
        price      = float(fast.last_price)
        prev_close = float(fast.previous_close)
        day_high   = float(fast.day_high)
        day_low    = float(fast.day_low)
        volume     = int(fast.three_month_average_volume or 0)
        week52_high= float(fast.year_high)
        week52_low = float(fast.year_low)
        mktcap     = float(fast.market_cap or 0)
    except Exception:
        hist       = ticker.history(period="5d")
        price      = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
        day_high   = float(hist["High"].iloc[-1])
        day_low    = float(hist["Low"].iloc[-1])
        volume     = int(hist["Volume"].iloc[-1])
        week52_high= float(hist["High"].max())
        week52_low = float(hist["Low"].min())
        mktcap     = 0.0

    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
    return {
        "price":        round(price, 2),
        "prev_close":   round(prev_close, 2),
        "change_pct":   round(change_pct, 2),
        "day_high":     round(day_high, 2),
        "day_low":      round(day_low, 2),
        "volume":       volume,
        "week52_high":  round(week52_high, 2),
        "week52_low":   round(week52_low, 2),
        "market_cap":   mktcap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_engineer(ticker_symbol: str, period: str = DATA_PERIOD) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance and compute 45+ technical features.
    Returns clean DataFrame (NaN rows dropped).
    """
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── Trend ─────────────────────────────────────────────────────────────────
    df["SMA_10"]       = ta.trend.sma_indicator(c, 10)
    df["SMA_20"]       = ta.trend.sma_indicator(c, 20)
    df["SMA_50"]       = ta.trend.sma_indicator(c, 50)
    df["EMA_9"]        = ta.trend.ema_indicator(c, 9)
    df["EMA_12"]       = ta.trend.ema_indicator(c, 12)
    df["EMA_26"]       = ta.trend.ema_indicator(c, 26)
    macd_obj           = ta.trend.MACD(c)
    df["MACD"]         = macd_obj.macd()
    df["MACD_signal"]  = macd_obj.macd_signal()
    df["MACD_hist"]    = macd_obj.macd_diff()
    df["ADX"]          = ta.trend.adx(h, l, c, window=14)
    df["CCI"]          = ta.trend.cci(h, l, c, window=20)
    aroon              = ta.trend.AroonIndicator(h, l, window=25)
    df["Aroon_up"]     = aroon.aroon_up()
    df["Aroon_down"]   = aroon.aroon_down()

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["RSI_14"]       = ta.momentum.rsi(c, window=14)
    df["RSI_7"]        = ta.momentum.rsi(c, window=7)
    stoch              = ta.momentum.StochasticOscillator(h, l, c)
    df["Stoch_K"]      = stoch.stoch()
    df["Stoch_D"]      = stoch.stoch_signal()
    df["Williams_R"]   = ta.momentum.williams_r(h, l, c, lbp=14)
    df["ROC"]          = ta.momentum.roc(c, window=12)

    # ── Volatility ────────────────────────────────────────────────────────────
    bb                 = ta.volatility.BollingerBands(c, window=20)
    df["BB_upper"]     = bb.bollinger_hband()
    df["BB_middle"]    = bb.bollinger_mavg()
    df["BB_lower"]     = bb.bollinger_lband()
    df["BB_width"]     = bb.bollinger_wband()
    df["BB_pct"]       = bb.bollinger_pband()
    df["ATR"]          = ta.volatility.average_true_range(h, l, c, window=14)
    kc                 = ta.volatility.KeltnerChannel(h, l, c, window=20)
    df["Keltner_upper"]= kc.keltner_channel_hband()
    df["Keltner_lower"]= kc.keltner_channel_lband()

    # ── Volume ────────────────────────────────────────────────────────────────
    df["OBV"]          = ta.volume.on_balance_volume(c, v)
    df["VWAP"]         = ta.volume.volume_weighted_average_price(h, l, c, v)
    df["MFI"]          = ta.volume.money_flow_index(h, l, c, v, window=14)
    df["CMF"]          = ta.volume.chaikin_money_flow(h, l, c, v, window=20)

    # ── Price-Derived ─────────────────────────────────────────────────────────
    df["HL_pct"]            = (h - l) / l
    df["CO_pct"]            = (c - df["Open"]) / df["Open"]
    df["Return_1d"]         = c.pct_change(1)
    df["Return_3d"]         = c.pct_change(3)
    df["Return_5d"]         = c.pct_change(5)
    df["Return_10d"]        = c.pct_change(10)
    df["Log_Return"]        = np.log(c / c.shift(1))
    df["Price_SMA20_ratio"] = c / df["SMA_20"]
    df["Price_SMA50_ratio"] = c / df["SMA_50"]
    df["Volatility_20d"]    = df["Return_1d"].rolling(20).std()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LABEL GENERATION  (multi-horizon)
# ─────────────────────────────────────────────────────────────────────────────

def make_labels(df: pd.DataFrame, horizons: list = [1, 5, 10, 30]) -> dict:
    """
    For each horizon h, generate:
      - y_price_h : future close price (regression target)
      - y_trend_h : 0=Up, 1=Down, 2=Neutral (classification target)
    Returns dict keyed by horizon.
    """
    close = df["Close"].values
    labels = {}
    for h in horizons:
        future_price = np.roll(close, -h).astype(float)
        future_price[-h:] = np.nan
        ret = (future_price - close) / close
        trend = np.where(ret > TREND_THRESHOLD, 0,
                np.where(ret < -TREND_THRESHOLD, 1, 2)).astype(int)
        labels[h] = {"price": future_price, "trend": trend, "ret": ret}
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WALK-FORWARD SPLIT  (no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_split(n: int):
    """
    Returns (train_end, val_end) indices for walk-forward split.
    Test set is always the last TEST_SPLIT fraction — never seen during training.
    """
    train_end = int(n * TRAIN_SPLIT)
    val_end   = int(n * (TRAIN_SPLIT + VAL_SPLIT))
    return train_end, val_end


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SCALERS
# ─────────────────────────────────────────────────────────────────────────────

def fit_scalers(X: np.ndarray, y_close: np.ndarray):
    """Fit RobustScaler on training data only."""
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    scaler_X.fit(X)
    scaler_y.fit(y_close.reshape(-1, 1))
    return scaler_X, scaler_y


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """
    Sliding-window dataset with optional Gaussian noise augmentation.
    Supports multi-horizon targets.
    """
    def __init__(
        self,
        X_scaled:   np.ndarray,        # (N, features)
        sentiment:  np.ndarray,        # (N, 6)  enriched sentiment vector
        y_price:    np.ndarray,        # (N, H)  H = number of horizons
        y_trend:    np.ndarray,        # (N, H)
        seq_len:    int = SEQ_LEN,
        augment:    bool = False,
        noise_std:  float = 0.002,
    ):
        self.X         = X_scaled
        self.sentiment = sentiment
        self.y_price   = y_price
        self.y_trend   = y_trend
        self.seq_len   = seq_len
        self.augment   = augment
        self.noise_std = noise_std
        self.N         = len(X_scaled) - seq_len

    def __len__(self):
        return max(0, self.N)

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len].copy()
        if self.augment:
            x_seq += np.random.normal(0, self.noise_std, x_seq.shape).astype(np.float32)

        # Most-recent sentiment in the window
        sent = self.sentiment[idx + self.seq_len - 1]

        return (
            torch.FloatTensor(x_seq),
            torch.FloatTensor(sent),
            torch.FloatTensor(self.y_price[idx + self.seq_len]),
            torch.LongTensor(self.y_trend[idx + self.seq_len]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  HIGH-LEVEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(ticker_symbol: str, sentiment_series: np.ndarray, horizons: list = [1, 5, 10, 30]):
    """
    Full pipeline: fetch → engineer → label → scale → split → Dataset objects.
    Returns (ds_train, ds_val, ds_test, scaler_X, scaler_y, df, feature_cols)
    """
    df         = fetch_and_engineer(ticker_symbol)
    labels     = make_labels(df, horizons)
    max_h      = max(horizons)
    valid_n    = len(df) - max_h

    feature_cols = [c for c in df.columns
                    if c not in ["Return_1d", "Return_3d", "Return_5d",
                                 "Return_10d", "Log_Return"]]
    X_raw      = df[feature_cols].values[:valid_n].astype(np.float32)

    # Multi-horizon stacked targets
    y_price = np.stack([labels[h]["price"][:valid_n] for h in horizons], axis=1).astype(np.float32)
    y_trend = np.stack([labels[h]["trend"][:valid_n] for h in horizons], axis=1).astype(np.int64)

    # Align sentiment
    if len(sentiment_series) < valid_n:
        pad_len = valid_n - len(sentiment_series)
        sentiment_series = np.vstack([
            np.zeros((pad_len, sentiment_series.shape[1]), dtype=np.float32),
            sentiment_series
        ])
    sent_arr = sentiment_series[:valid_n].astype(np.float32)

    # Walk-forward split
    train_end, val_end = walk_forward_split(valid_n)
    scaler_X, scaler_y = fit_scalers(X_raw[:train_end], df["Close"].values[:train_end])

    X_sc  = scaler_X.transform(X_raw).astype(np.float32)
    yp_sc = scaler_y.transform(y_price[:, :1]).astype(np.float32)  # scale first horizon for reference
    # For regression targets keep raw (model predicts scaled, we inverse later)
    y_price_sc = np.hstack([
        scaler_y.transform(y_price[:, i:i+1]) for i in range(y_price.shape[1])
    ]).astype(np.float32)

    ds_train = StockSequenceDataset(
        X_sc[:train_end], sent_arr[:train_end],
        y_price_sc[:train_end], y_trend[:train_end],
        augment=True
    )
    ds_val   = StockSequenceDataset(
        X_sc[train_end:val_end], sent_arr[train_end:val_end],
        y_price_sc[train_end:val_end], y_trend[train_end:val_end],
        augment=False
    )
    ds_test  = StockSequenceDataset(
        X_sc[val_end:], sent_arr[val_end:],
        y_price_sc[val_end:], y_trend[val_end:],
        augment=False
    )

    return ds_train, ds_val, ds_test, scaler_X, scaler_y, df, feature_cols, X_sc, sent_arr
