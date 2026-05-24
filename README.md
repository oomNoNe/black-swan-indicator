# 🚨 Black Swan Risk Indicator (Quant Edition)

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
