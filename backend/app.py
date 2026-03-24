"""
app.py — Flask REST API Server

Endpoints:
  GET  /                      → serve frontend
  GET  /api/stocks            → list of NIFTY 50 companies
  GET  /api/realtime?company=TCS
  POST /api/predict
  GET  /api/history?company=TCS&period=6mo
  GET  /api/eval?company=TCS
  GET  /api/status?company=TCS
"""

import os
import json
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import yfinance as yf

from config import NIFTY50_TICKERS, MODEL_DIR
from data_pipeline import get_realtime_price
import torch

# FORCE all models to load on CPU
torch.set_default_tensor_type(torch.FloatTensor)

# Monkey patch torch.load to always use CPU
_original_load = torch.load

def cpu_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return _original_load(*args, **kwargs)

torch.load = cpu_load

# ─────────────────────────────────────────────────────────────
# App Init
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────
# Lazy Model Cache
# ─────────────────────────────────────────────────────────────

_model_cache = {}  # company → (models, meta)


def _is_trained(company: str) -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, f"{company}_ensemble.pkl"))


def _get_or_train(company: str, force: bool = False):
    """
    Load trained ensemble or trigger training.
    """
    if not force and company in _model_cache:
        return _model_cache[company]

    from orchestrator import run, is_trained

    if force or not is_trained(company):
        models, meta = run(company, retrain=force)
    else:
        from trainer import load_ensemble
        models, meta = load_ensemble(company)

    _model_cache[company] = (models, meta)
    return models, meta


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    return jsonify({"stocks": sorted(NIFTY50_TICKERS.keys())})


@app.route("/api/status", methods=["GET"])
def get_status():
    company = request.args.get("company", "").upper()

    if company not in NIFTY50_TICKERS:
        return jsonify({"error": "Unknown stock"}), 400

    trained = _is_trained(company)

    report_path = os.path.join(
        os.path.dirname(__file__),
        "reports",
        f"{company}_eval.json"
    )

    metrics = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            metrics = json.load(f).get("horizons", {})

    return jsonify({
        "company": company,
        "trained": trained,
        "in_cache": company in _model_cache,
        "metrics": metrics,
    })


@app.route("/api/realtime", methods=["GET"])
def get_realtime():
    company = request.args.get("company", "").upper()

    if company not in NIFTY50_TICKERS:
        return jsonify({"error": "Unknown stock"}), 400

    try:
        ticker = NIFTY50_TICKERS[company]
        rt = get_realtime_price(ticker)

        rt["company"] = company
        rt["ticker"] = ticker
        rt["fetched_at"] = datetime.now().isoformat()

        return jsonify(rt)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    company = data.get("company", "").upper()
    retrain = bool(data.get("retrain", False))

    if not company:
        return jsonify({"error": "company is required"}), 400

    if company not in NIFTY50_TICKERS:
        return jsonify({"error": f"Unknown stock: {company}"}), 400

    try:
        # Ensure model is trained
        _get_or_train(company, force=retrain)

        # Run prediction
        from predictor import predict as _predict
        result = _predict(company)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    company = request.args.get("company", "").upper()
    period = request.args.get("period", "6mo")

    if company not in NIFTY50_TICKERS:
        return jsonify({"error": "Unknown stock"}), 400

    try:
        ticker = NIFTY50_TICKERS[company]
        df = yf.Ticker(ticker).history(period=period)[
            ["Open", "High", "Low", "Close", "Volume"]
        ]

        df.index = df.index.strftime("%Y-%m-%d")

        return jsonify({
            "company": company,
            "ticker": ticker,
            "period": period,
            "history": df.to_dict(orient="index"),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/eval", methods=["GET"])
def get_eval():
    company = request.args.get("company", "").upper()

    report_path = os.path.join(
        os.path.dirname(__file__),
        "reports",
        f"{company}_eval.json"
    )

    if not os.path.exists(report_path):
        return jsonify({
            "error": "No evaluation report found. Train the model first."
        }), 404

    with open(report_path) as f:
        return jsonify(json.load(f))


# ─────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(
    os.path.dirname(__file__),
    "../frontend"
)

@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ─────────────────────────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 StockSense AI Stock Engine starting on http://localhost:5000")
    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000,
        threaded=True
    )

