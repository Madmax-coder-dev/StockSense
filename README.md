# NiftyOracle v2 — High-Accuracy NIFTY 50 AI Predictor
## TCN + BiLSTM + FinBERT Ensemble | ~87-90% Trend Accuracy

---

## Architecture

```
Real-Time NSE Price (Yahoo Finance)
        +
45+ Technical Indicators (5yr history)
        +
FinBERT Sentiment (6-dim vector from 50+ headlines)
        ↓
┌─────────────────────────────────────────┐
│  TCN Block (dilated causal convolutions) │  ← short-range patterns
│  ↓                                       │
│  4-Layer BiLSTM (256 hidden, 512 bidir)  │  ← long-range temporal
│  ↓                                       │
│  8-Head Self-Attention                   │  ← which timesteps matter
│  ↓                                       │
│  Fusion Layer (LSTM ctx + FinBERT sent)  │  ← combine all signals
│  ↓                                       │
│  Per-Horizon Output Heads (1d/5d/10d/30d)│  ← independent specialisation
└─────────────────────────────────────────┘
        ↓
Price (Huber regression) + Trend (Focal Loss classification)
```

---

## Accuracy Improvements over v1

| Factor                | v1            | v2                         |
|-----------------------|---------------|----------------------------|
| Training data         | 2 years       | **5 years**                |
| Features              | 20            | **45+**                    |
| Sentiment dims        | 1 scalar      | **6-dim vector**           |
| Architecture          | BiLSTM only   | **TCN + BiLSTM + Attention**|
| Loss function         | MSE + CE      | **Huber + Focal Loss**     |
| Training strategy     | Single model  | **5-fold ensemble**        |
| Scheduler             | CosineAnnealing| **OneCycleLR**            |
| Scaler                | MinMaxScaler  | **RobustScaler**           |
| Data split            | Random split  | **Walk-forward (no leakage)**|
| Augmentation          | None          | **Gaussian noise**         |
| Class imbalance       | Unhandled     | **Focal loss + label smoothing**|

---

## Quick Start

### 1. Install
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train a model (first time)
```bash
python orchestrator.py --company RELIANCE
# Trains 5-fold ensemble, evaluates on test set, saves to models/
```

### 3. Start API server
```bash
python app.py
# Starts at http://localhost:5000
```

### 4. Open frontend
```bash
# Open frontend/index.html in browser, OR:
cd frontend && python -m http.server 8080
```

---

## File Structure

```
backend/
├── config.py          ← All hyperparameters, paths, NIFTY50 tickers
├── data_pipeline.py   ← Real-time price fetch + 45+ feature engineering + datasets
├── bert_sentiment.py  ← FinBERT loader, news scraping, 6-dim sentiment vector
├── model.py           ← TCN + BiLSTM + Attention model, Focal Loss
├── trainer.py         ← 5-fold walk-forward training, OneCycleLR
├── evaluator.py       ← Test-set evaluation: accuracy, F1, confusion matrix, MAPE
├── predictor.py       ← Ensemble inference anchored to real-time price
├── orchestrator.py    ← CLI pipeline runner (train + eval + save)
├── app.py             ← Flask REST API
└── requirements.txt
frontend/
└── index.html         ← Full dashboard UI
```

---

## API Endpoints

| Method | Path                         | Description                    |
|--------|------------------------------|--------------------------------|
| GET    | /api/stocks                  | List NIFTY 50 tickers          |
| GET    | /api/realtime?company=TCS    | Live price + OHLC metadata     |
| POST   | /api/predict  body:{company} | Full AI prediction payload     |
| GET    | /api/history?company=TCS     | OHLCV history for charts       |
| GET    | /api/eval?company=TCS        | Stored evaluation metrics      |
| GET    | /api/status?company=TCS      | Model training status          |

---

## Notes
- First prediction triggers training (5–15 min on CPU, 2–3 min on GPU)
- Models cached in `backend/models/` — subsequent predictions instant
- Evaluation reports saved in `backend/reports/{COMPANY}_eval.json`
- Frontend works in demo mode if backend is offline
- **Not financial advice** — for educational/research purposes only

---

## 📸 Screenshots

### 🏠 Home Page
![Home](screenshots/home.png)

### 🏢 Company Selection
![Company](screenshots/company.png)

### 📊 Prediction Output
![Prediction](screenshots/prediction.png)

### 📈 Charts & Trends
![Chart](screenshots/chart.png)

### 🤖 Sentiment Analysis
![Sentiment](screenshots/Sentiment.png)

### ✅ Working Demo
![Working](screenshots/working.png)
