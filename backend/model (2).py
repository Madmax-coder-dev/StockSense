"""
model.py — Hybrid Deep Learning Model Architecture

Design for high accuracy:
  ┌─────────────────────────────────────────────────────────────┐
  │  INPUT (60 days × 45 features)                              │
  │  ↓                                                          │
  │  TCN Block  (Temporal Convolutional Network)                │
  │    → captures short-range patterns with dilated convolutions│
  │  ↓                                                          │
  │  BiLSTM Stack  (4 layers, 256 hidden, bidirectional)        │
  │    → captures long-range temporal dependencies              │
  │  ↓                                                          │
  │  Multi-Head Self-Attention  (8 heads)                       │
  │    → learns which time steps matter most                    │
  │  ↓                                                          │
  │  Fusion Layer                                               │
  │    → concatenates LSTM context + 6-dim FinBERT sentiment    │
  │  ↓                                                          │
  │  Multi-Horizon Output Heads  (1d / 5d / 10d / 30d)         │
  │    → Price regression  +  Trend classification per horizon  │
  └─────────────────────────────────────────────────────────────┘

  Key accuracy levers:
    • TCN front-end extracts local momentum patterns
    • Residual connections prevent gradient vanishing
    • Layer normalisation stabilises training
    • Focal loss combats Up/Down/Neutral imbalance
    • Per-horizon independent heads (not shared decoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import (
    LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    ATTN_HEADS, SENTIMENT_DIM, DEVICE,
    USE_FOCAL_LOSS, LABEL_SMOOTHING
)

HORIZONS = [1, 5, 10, 30]
N_CLASSES = 3   # Up, Down, Neutral


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class CausalConvBlock(nn.Module):
    """Dilated causal convolution with residual connection (TCN building block)."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(0.1)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x: (B, C, T)
        res = self.res(x)
        out = F.gelu(self.norm1(self.conv1(x)[:, :, :x.size(2)]))
        out = self.drop(out)
        out = F.gelu(self.norm2(self.conv2(out)[:, :, :x.size(2)]))
        return out + res


class TCNEncoder(nn.Module):
    """3-layer TCN with exponentially increasing dilation."""
    def __init__(self, in_features: int, channels: int = 128):
        super().__init__()
        self.blocks = nn.Sequential(
            CausalConvBlock(in_features, channels, kernel=3, dilation=1),
            CausalConvBlock(channels, channels, kernel=3, dilation=2),
            CausalConvBlock(channels, channels, kernel=3, dilation=4),
        )

    def forward(self, x):  # x: (B, T, F)
        return self.blocks(x.permute(0, 2, 1)).permute(0, 2, 1)  # → (B, T, C)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for the attention layer."""
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):  # x: (B, T, D)
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pe(pos)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class HybridStockModel(nn.Module):
    """
    TCN → BiLSTM → Multi-Head Attention → Fusion → Multi-Horizon Heads
    """
    TCN_DIM = 128

    def __init__(self, input_size: int, horizons: list = HORIZONS):
        super().__init__()
        self.horizons = horizons
        H = LSTM_HIDDEN
        h = len(horizons)

        # TCN front-end
        self.tcn = TCNEncoder(input_size, self.TCN_DIM)

        # BiLSTM stack
        self.lstm = nn.LSTM(
            input_size=self.TCN_DIM,
            hidden_size=H,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )

        # Multi-head self-attention
        self.pos_enc  = PositionalEncoding(H * 2)
        self.attn     = nn.MultiheadAttention(H * 2, num_heads=ATTN_HEADS,
                                               batch_first=True, dropout=0.1)
        self.attn_norm = nn.LayerNorm(H * 2)

        # Sentiment branch
        self.sent_net = nn.Sequential(
            nn.Linear(SENTIMENT_DIM, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.GELU(),
        )

        # Fusion
        fuse_in = H * 2 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
        )

        # Per-horizon heads (independent — better specialisation)
        self.price_heads = nn.ModuleList([nn.Linear(256, 1) for _ in horizons])
        self.trend_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, N_CLASSES)
            ) for _ in horizons
        ])

    def forward(self, x: torch.Tensor, sentiment: torch.Tensor):
        """
        x:         (B, T, features)
        sentiment: (B, 6)
        Returns:
          prices: (B, H)   — scaled price predictions per horizon
          trends: (B, H, 3) — logits per horizon
        """
        # TCN
        tcn_out = self.tcn(x)                        # (B, T, 128)

        # BiLSTM
        lstm_out, _ = self.lstm(tcn_out)             # (B, T, 2H)

        # Self-attention on LSTM output
        pe_out    = self.pos_enc(lstm_out)
        attn_out, _ = self.attn(pe_out, pe_out, pe_out)
        ctx       = self.attn_norm(lstm_out + attn_out)[:, -1, :]  # last step

        # Sentiment
        sent_feat = self.sent_net(sentiment)          # (B, 64)

        # Fusion
        fused = self.fusion(torch.cat([ctx, sent_feat], dim=-1))  # (B, 256)

        # Heads
        prices = torch.cat([h(fused) for h in self.price_heads], dim=-1)   # (B, H)
        trends = torch.stack([h(fused) for h in self.trend_heads], dim=1)  # (B, H, 3)

        return prices, trends


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FOCAL LOSS  (handles class imbalance — critical for Neutral-heavy data)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """logits: (B, C), targets: (B,) int64"""
        # Label smoothing
        n_cls = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.ls / (n_cls - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)

        log_p  = F.log_softmax(logits, dim=-1)
        p      = torch.exp(log_p)
        focal  = (1 - p) ** self.gamma
        loss   = -(smooth_targets * focal * log_p).sum(dim=-1)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COMBINED LOSS
# ─────────────────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Weighted combination of:
      • Huber loss for price regression  (robust to outliers)
      • Focal loss for trend classification
    """
    def __init__(self, price_w: float, trend_w: float):
        super().__init__()
        self.price_w   = price_w
        self.trend_w   = trend_w
        self.huber     = nn.HuberLoss(delta=0.5)
        self.focal     = FocalLoss()

    def forward(self, pred_prices, pred_trends, true_prices, true_trends):
        """
        pred_prices:  (B, H)
        pred_trends:  (B, H, 3)
        true_prices:  (B, H)
        true_trends:  (B, H) int64
        """
        B, H = pred_prices.shape

        # Price loss (averaged across horizons)
        price_loss = self.huber(pred_prices, true_prices)

        # Trend loss (averaged across horizons, focal per class)
        trend_loss = 0.0
        for i in range(H):
            trend_loss += self.focal(pred_trends[:, i, :], true_trends[:, i])
        trend_loss /= H

        return self.price_w * price_loss + self.trend_w * trend_loss, price_loss, trend_loss
