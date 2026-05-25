# 🚨 Black Swan Risk Indicator (Quant Edition)

> 🌐 **Languages:** **English** | [ภาษาไทย](README.th.md)

[![Live Report](https://img.shields.io/badge/📊_Live_Report-View_now-success?style=for-the-badge)](https://oomNoNe.github.io/black-swan-indicator/)

[![CI](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 👉 **Click the green badge above** to see the live report instantly (no install required)

AI-powered & Quant-driven early warning system for financial crises.
Combines **NLP sentiment analysis**, **macro volatility forecasting**, and **regime detection**
to produce a single Crisis Risk Score (0–100) and a backtested trading strategy.

---

## ✨ Features

| Module | Tech | What it does |
|---|---|---|
| **Live Risk Dashboard** | Streamlit + Plotly | Real-time VIX, FinBERT sentiment on Google News, weighted Crisis Score gauge |
| **AI Regime & Prediction** | XGBoost Regressor | 7-day VIX forecast from lagged macro features |
| **Quant Backtest** | NumPy / Pandas | Sharpe, Max Drawdown, Win Rate, Profit Factor vs. Buy & Hold |
| **Regime Detector** | SMA + Rolling Vol | Classifies market as Trending Bull / Ranging / Panic and dynamically reweights the risk equation |

---

## 🧮 The Crisis Equation

```
Crisis_Score = w_news(regime) × News_Risk + w_market(regime) × Market_Risk
```

Weights adapt to regime:

| Regime | w_news | w_market |
|---|---|---|
| Trending Bull | 0.3 | 0.7 |
| Panic | 0.7 | 0.3 |
| Ranging / Unknown | 0.5 | 0.5 |

Backtest signal triggers when **VIX > 30 AND 20-day rolling vol > 1.5× 252-day median vol**.

---

## 🗂️ Project Structure

```
black-swan-indicator/
├── app.py                    # Streamlit entry point (3 tabs)
├── data/
│   ├── market_crawler.py     # yfinance: S&P 500 + VIX
│   └── news_crawler.py       # Google News RSS scraper
├── engine/
│   ├── ai_model.py           # FinBERT sentiment wrapper
│   ├── ml_predictor.py       # XGBoost VIX forecaster
│   ├── regime_detector.py    # Market regime + dynamic risk equation
│   └── backtester.py         # Sharpe / MDD / Win Rate / Profit Factor
└── ui/
    └── components.py         # Plotly gauge + backtest chart
```

---

## 🚀 Quick Start

### Option A — Local Python
```bash
git clone https://github.com/oomNoNe/black-swan-indicator.git
cd black-swan-indicator
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Option B — Docker (recommended for deployment)
```bash
docker build -t black-swan-indicator .
docker run -p 8501:8501 black-swan-indicator
```
Open http://localhost:8501

The first run downloads the FinBERT model (~440 MB). Subsequent runs are cached.

### Option C — Streamlit Cloud (one-click hosted)
1. Fork this repo to your GitHub account
2. Go to https://share.streamlit.io
3. Click **"New app"** → select your fork → main branch → `app.py`
4. Deploy and get a public URL

---

## 🧪 Development

Run tests:
```bash
pytest tests/ -v
```

CI runs automatically on push via GitHub Actions ([workflow](.github/workflows/ci.yml)).

---

## 🎓 Why These Models?

### 🤖 FinBERT (for Sentiment Analysis)

**Rationale**: Pre-trained on financial corpora — understands jargon that general
language models miss (e.g., "bearish guidance", "hawkish Fed", "deleveraging").

| ✅ Pros | ❌ Cons |
|---|---|
| ~15% more accurate than vanilla BERT on financial text | English only |
| Open-source, free (ProsusAI) | Trained 2019 — doesn't know new jargon |
| CPU-friendly inference | ~440MB memory footprint |
| Pre-trained on Reuters TRC2 | Possible bias from training data |

**Alternatives considered**: GPT-4 (too expensive, latency), generic BERT (no
domain context), VADER (rule-based, not ML).

### 🌲 XGBoost (for VIX Forecasting)

**Rationale**: Best-in-class for tabular data, which is what our macro lag
features look like. Battle-tested in Kaggle and used by major hedge funds.

| ✅ Pros | ❌ Cons |
|---|---|
| Industry-proven (Kaggle, quant funds) | No temporal memory — manual lag features needed |
| Extremely fast (< 1s training) | Easy to overfit without tuning |
| Interpretable via feature importance | Poor extrapolation |
| Handles missing values natively | Outperformed by deep learning on huge datasets |
| No feature scaling required | |

**Alternatives considered**: Linear regression (misses non-linearity), LSTM
(data-hungry, slow, opaque), ARIMA (assumes linearity/stationarity), Transformer
(overkill for 5 features).

### 📊 SMA + Rolling Vol (for Regime Detection)

**Rationale**: Simple, transparent, and used as a standard by major funds
(Bridgewater, Two Sigma). No ML required — Occam's Razor applied.

| ✅ Pros | ❌ Cons |
|---|---|
| Transparent — not a black box | **Lagging indicator** — SMA-200 reacts slowly |
| No hyperparameter tuning | Binary thresholds miss transitions |
| Fast computation | Doesn't use macro/news data |
| Industry standard | |

**Future alternatives**: Hidden Markov Model (smoother transitions),
Bayesian regime switching (probabilistic), GMM clustering (unsupervised).

---

## 📊 Methodology Notes

- **Time-series split** (`shuffle=False`) for XGBoost training — no look-ahead leakage.
- **Lag features**: VIX lag 1/3/7 days, S&P returns 1d/5d.
- **Sentiment**: ProsusAI/finbert, truncated to 512 tokens, mapped to 0/50/100.
- **Caching**: `@st.cache_resource` for models, `@st.cache_data(ttl=1h)` for market data.

---

## 🔭 Roadmap

**Tier 1 — Deploy & Polish**
- [x] Dockerfile (multi-stage, non-root)
- [x] GitHub Actions CI (pytest + lint)
- [x] CI/license/Python badges
- [ ] Streamlit Cloud live demo URL

**Tier 2 — ML/Quant Upgrades** ✅ DONE
- [x] Walk-forward validation (TimeSeriesSplit) replacing single train/test
- [x] Macro features: yield curve spread, Gold, Oil, DXY (5 → 13 features)
- [x] Classification mode: crash vs no-crash + Precision/Recall/F1/ROC-AUC
- [x] SHAP feature importance (interpretable AI)
- [x] Transaction cost in backtest (configurable, 0-50 bps)
- [x] Model comparison: XGBoost vs LightGBM vs Ridge/LogReg (LSTM in Tier 3)

**Tier 3 — Production** ✅ 4/6 DONE
- [ ] Batch pipeline (Airflow/Prefect) — *skipped: GitHub Actions cron is lighter for portfolio*
- [ ] PostgreSQL/DuckDB store — *skipped: in-memory caching sufficient at this scale*
- [x] Discord webhook alerts (configurable threshold + test button)
- [x] Multi-asset expansion (10 assets: US equity, EM, crypto, commodities)
- [x] LSTM model added to comparison (PyTorch)
- [x] MLflow experiment tracking (local file store)

---

## ⚠️ Disclaimer

For educational and research purposes only. Not financial advice.
