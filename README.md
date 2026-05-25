# 🚨 Black Swan Risk Indicator (Quant Edition)

> 🌐 **Languages:** **English** | [ภาษาไทย](README.th.md)

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

```bash
# 1. Clone & setup
git clone <repo-url>
cd black-swan-indicator
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

The first run downloads the FinBERT model (~440 MB). Subsequent runs are cached.

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

- [ ] Walk-forward validation instead of single train/test split
- [ ] Add yield curve inversion, credit spreads, gold/oil features
- [ ] Switch to classification (crash vs. no-crash) with Precision/Recall
- [ ] SHAP feature importance overlay
- [ ] Discord/Line webhook alert when score > 70
- [ ] Dockerfile + Streamlit Cloud deploy

---

## ⚠️ Disclaimer

For educational and research purposes only. Not financial advice.
