# 🚨 Black Swan Risk Indicator

> 🌐 **Languages:** **English** | [ภาษาไทย](README.th.md)

[![Live Report (EN)](https://img.shields.io/badge/📊_Live_Report-English-success?style=for-the-badge)](https://oomNoNe.github.io/black-swan-indicator/index.en.html)
[![Live Report (TH)](https://img.shields.io/badge/📊_รายงานสด-ภาษาไทย-blue?style=for-the-badge)](https://oomNoNe.github.io/black-swan-indicator/)

[![CI](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

**Black Swan Risk Indicator** is an early-warning system for financial market crises.
It combines **NLP sentiment analysis** of global financial news with **time-series ML
forecasting** and **rule-based regime detection** to produce a single **Crisis Risk
Score (0–100)** — backed by a 5-year historical backtest.

The system surfaces when markets are likely to enter a "fear regime" so investors
can prepare for downside scenarios like the 2008 crisis, COVID-2020 crash, or
inflation shock of 2022.

🌐 **[View Live Report →](https://oomNoNe.github.io/black-swan-indicator/)** · Built with Python · MIT License

---

## 🔥 The Problem

Most retail investors are **caught off guard** by market crashes because the tools
they use are backward-looking:

- 📊 **Stock screeners** show fundamentals — useless when markets are panicking
- 📈 **Technical indicators** lag — they confirm crashes *after* they start
- 📰 **News feeds** are reactive — by the time a crash makes headlines, it's too late
- 🤖 **AI tools** focus on returns prediction — but volatility (fear) is what kills portfolios

Meanwhile, **professional quants** use ensembles of macro indicators + sentiment
analysis + regime models that aren't accessible to most people.

**This project asks**: *Can we build a transparent, open-source crisis detector
that an individual investor can run on their laptop?*

---

## 💡 The Solution

A multi-layer detection pipeline that fuses **3 independent signals**:

1. 📰 **News sentiment** (FinBERT analyzes global financial headlines)
2. 📊 **Market volatility** (VIX + 20-day realized vol + threshold rules)
3. 🤖 **ML forecasting** (XGBoost on 13 macro features predicts 7-day VIX)

These feed into a dynamically-weighted **Crisis Equation** that adapts to the
current market regime (Bull / Panic / Ranging).

When the score crosses **70**, the system can fire a **Discord webhook alert**
so you can act before headlines catch up.

---

## 🗺️ How It Works

```
Daily / On-demand:

┌─────────────────────────────────────────────────────────────┐
│  1. Fetch data                                              │
│     yfinance: S&P 500, VIX, 10Y/3M Treasury, Gold, Oil,     │
│     DXY (~1,200 trading days, 8 series)                     │
│     Google News RSS: 10 latest financial headlines          │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Feature engineering                                     │
│     13 features: VIX lags (1/3/7d), momentum, S&P           │
│     returns (1d/5d), realized vol, yield curve spread,      │
│     inversion flag, gold/oil/DXY momentum                   │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Three parallel pipelines                                │
│     ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│     │ FinBERT     │  │ XGBoost      │  │ Regime          │  │
│     │ sentiment   │  │ 7-day VIX    │  │ classifier      │  │
│     │ (NEG/NEU/   │  │ forecast +   │  │ (SMA-50/200     │  │
│     │  POS)       │  │ walk-forward │  │  + rolling vol) │  │
│     └─────────────┘  └──────────────┘  └─────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Crisis Equation (regime-weighted fusion)                │
│     Score = w_news(regime)   × NewsRisk                     │
│           + w_market(regime) × MarketRisk                   │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Outputs                                                 │
│     • Crisis Risk Score (0–100) on gauge                    │
│     • Backtest equity curve + drawdown vs Buy & Hold        │
│     • Multi-asset volatility comparison                     │
│     • Discord alert if Score > threshold (configurable)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Presentation Layer                                              │
│  ├── Streamlit app (5 tabs, interactive sliders)                 │
│  └── Static HTML report (auto-rebuilt weekly via cron)           │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Engine Layer (engine/)                                          │
│  ├── features.py            — 13-feature builder                 │
│  ├── ml_predictor.py        — 5 models (XGB, LGBM, Ridge,        │
│  │                             LSTM, Naive baseline)             │
│  ├── ai_model.py            — FinBERT sentiment wrapper          │
│  ├── regime_detector.py     — Market mood classifier             │
│  ├── backtester.py          — Sharpe / MDD / transaction cost    │
│  ├── alerts.py              — Discord webhook                    │
│  ├── experiment_tracker.py  — MLflow integration                 │
│  └── disk_cache.py          — joblib + parquet persistence       │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Data Layer (data/)                                              │
│  ├── market_crawler.py      — yfinance (10 assets supported)     │
│  └── news_crawler.py        — Google News RSS                    │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Infrastructure                                                  │
│  ├── GitHub Actions         — CI/CD + weekly cron rebuild        │
│  ├── GitHub Pages           — Free static hosting                │
│  ├── Docker                 — Containerized deployment           │
│  └── Streamlit Cloud        — Optional managed hosting           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Findings (Honest Results)

This project documents **intellectual honesty over hype**.

### 🏆 Finding 1: Naive baseline beats every ML model
After tuning 4 ML models (XGBoost, LightGBM, Ridge, LSTM) with walk-forward
validation, the **persistence baseline** (`VIX in 7 days = VIX today`) won:

| Model | Mean R² (walk-forward CV, 5 folds) |
|---|---|
| 🥇 **Naive (baseline)** | **+0.093** |
| 🥈 XGBoost (tuned, regularized) | −0.122 |
| 🥉 LightGBM | −0.184 |
| Ridge | −0.131 |
| LSTM (PyTorch) | −3.720 |

**Interpretation**: VIX contains strong random-walk behavior. This validates
**Fama's Efficient Market Hypothesis** on the volatility-of-volatility surface.

🎓 **Engineering lesson**: Always test naive baselines first. If your fancy
ML doesn't beat persistence, **don't ship the ML**.

### 🦠 Finding 2: Rule-based system worked during COVID-2020
Despite poor ML forecasting, the rule-based detector (`VIX > 30 AND
vol_spike > 1.5×`) flagged the COVID crash on **March 9, 2020** — **7 days
before** VIX peaked at 82.69 on March 16.

An investor who acted on the signal could have avoided a ~30% drawdown.

### 💰 Finding 3: Transaction costs matter
Without trading costs, the strategy looks great. With realistic 10 bps per
turnover, Sharpe drops by ~0.05 — small but meaningful. **Many academic
backtests omit this** and overstate strategy performance.

---

## 🎓 Research Methodology

Structured as a **comparative study** answering: *Does news sentiment +
ML forecasting + rule-based regime detection beat Buy & Hold on a
risk-adjusted basis?*

| Step | Decision |
|---|---|
| **H₀ / H₁** | `Sharpe_strategy ≤ Sharpe_BH` vs `Sharpe_strategy > Sharpe_BH` |
| **α** | 0.05 · walk-forward k=5 · all ML must beat Naive baseline |
| **Test stats** | R², Sharpe Δ, Max Drawdown, Win Rate, Profit Factor, COVID lead-time |
| **Data** | 1,259 trading days · 8 macro series · 10 bps txn cost |
| **Conclusion** | ❌ ML fails to beat Naive · ✅ Rule-based detector caught COVID 7d early · ⚠️ Defensive strategy underperforms in bull markets |

<details>
<summary>📖 Full 5-step methodology (click to expand)</summary>

### 1. Hypothesis
- **H₀ (Null)**: The Black Swan strategy performs no better than Buy & Hold
  on a risk-adjusted basis: `Sharpe_strategy ≤ Sharpe_BH`
- **H₁ (Alternative)**: The Black Swan strategy delivers superior risk-adjusted
  returns: `Sharpe_strategy > Sharpe_BH`

### 2. Significance Level (α)
- **α = 0.05** for any formal tests (standard academic threshold)
- Walk-forward cross-validation with **k = 5 folds** to reduce single-split bias
- All ML forecasting compared against **Naive persistence baseline**
  (must beat baseline to be considered useful)

### 3. Test Statistics
| Question | Metric used |
|---|---|
| Forecast accuracy | **R² (walk-forward mean ± std)** vs Naive baseline |
| Strategy outperformance | **Sharpe Ratio** delta (strategy − B&H) |
| Tail risk reduction | **Max Drawdown** comparison |
| Trade quality | **Win Rate** + **Profit Factor** |
| Crisis detection | Lead time before market peak (case study) |

### 4. Calculation
- **Data**: ~1,259 trading days (5 years), 8 macro series (yfinance)
- **Backtest realism**: transaction cost = 10 bps per turnover
- **Validation**: TimeSeriesSplit (no shuffle, prevents look-ahead)
- **Replication**: cached models in `.cache/` for reproducibility

### 5. Conclusion
- ❌ **ML forecasting fails to beat Naive baseline** (R² ≈ 0.09 for Naive,
  negative for all ML models). Fails to reject H₀ for predictive task.
- ✅ **Rule-based crisis detector validated on COVID-19 case study**
  (signal fired 7 days before VIX peaked at 82.69).
- ⚠️ **In bull markets, defensive strategy underperforms Buy & Hold** — the
  value of crash-avoidance only shows during actual crises (insufficient
  sample size for formal test).

**Future statistical work**: Diebold-Mariano test for forecast accuracy,
bootstrap CI for Sharpe difference, paired t-test on daily returns.

</details>

---

## 🧠 Engineering Challenges

5 non-obvious trade-offs from building this:

| # | Challenge | One-line takeaway |
|---|---|---|
| 1 | Walk-forward vs naive split | Chose 5-10× slower CV for honest results |
| 2 | LSTM on small data | Documented R² = -3.72; kept for breadth |
| 3 | Static vs live Streamlit | Hybrid: HTML for share, Streamlit for dev |
| 4 | Slow repeat builds | Disk cache 7× speedup (60s → 8.7s) |
| 5 | FinBERT vocab drift | Accepted limit vs paid GPT-4 API |

<details>
<summary>📖 Read each in detail (full reasoning + alternatives considered)</summary>

### 1. Choosing the right validation strategy
**Problem**: Random `train_test_split` leaked future information into training,
inflating R² to fake-good levels.

**Solution**: Switched to `TimeSeriesSplit` with **walk-forward expanding
window**. Train on [0..t], test on [t..t+k], roll forward. Same as production.

**Trade-off**: 5-10× slower than single split, but the only honest way to
evaluate time-series ML.

### 2. Small dataset for deep learning
**Problem**: ~1,200 trading days is *tiny* for LSTM. It overfit dramatically
(R² of −3.72 vs Ridge's −0.13).

**Solution**: Documented this honestly in the report. Kept LSTM in the
comparison to show breadth, but recommended Ridge/Naive for production.
**Lesson**: deep learning isn't always the answer.

### 3. Static report vs live Streamlit trade-off
**Problem**: Streamlit Cloud free tier (1 GB RAM) couldn't host FinBERT
(~440 MB) + XGBoost + multi-asset fetch. Cold starts took 5+ minutes.

**Solution**: Hybrid architecture:
- **Static HTML report** (auto-rebuilt weekly) → fast share with anyone
- **Streamlit app** → run locally for interactive deep-dives

The static report serves 99% of viewers in < 1s. The Streamlit app stays
for the 1% who want to tweak parameters.

### 4. Persistent disk cache for repeated builds
**Problem**: Each `build_report.py` run took ~60s (fetch yfinance, train
XGBoost, walk-forward × 4 models, SHAP). Iterating on UI was painful.

**Solution**: Built `engine/disk_cache.py` — TTL-based parquet + joblib
caching. 2nd run drops to **~8.7s (7× speedup)**.

### 5. Sentiment scoring drift
**Problem**: FinBERT was trained in 2019. New jargon ("AI bubble",
"GameStop saga") falls outside its vocabulary.

**Trade-off accepted**: Documented limitation in the glossary. Alternatives
(GPT-4 API) are expensive and add latency. For an open-source educational
project, FinBERT remains the right pick.

</details>

---

## 🚀 Quick Start

### Option A — Just view the live report
👉 **[oomNoNe.github.io/black-swan-indicator/](https://oomNoNe.github.io/black-swan-indicator/)** (no install required)

### Option B — Local Python install
```bash
git clone https://github.com/oomNoNe/black-swan-indicator.git
cd black-swan-indicator
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate static report (8.7s after first run, ~60s first time)
python scripts/build_report.py
start docs/index.html

# OR run interactive Streamlit app
streamlit run app.py
```

### Option C — Docker
```bash
docker build -t black-swan-indicator .
docker run -p 8501:8501 black-swan-indicator
```

---

## 🗂️ Project Structure

4 layers: `data/` → `engine/` → `ui/` + `scripts/` for batch jobs.
See [Architecture](#-architecture) above for the dependency diagram.

<details>
<summary>📂 Full file tree (click to expand)</summary>

```
black-swan-indicator/
├── app.py                       # Streamlit entry (5 tabs)
├── scripts/
│   └── build_report.py          # Static HTML report generator
├── data/
│   ├── market_crawler.py        # yfinance + macro features
│   └── news_crawler.py          # Google News RSS scraper
├── engine/
│   ├── features.py              # Feature engineering (13 features)
│   ├── ml_predictor.py          # Model registry + walk-forward CV
│   ├── lstm_model.py            # PyTorch LSTM with sklearn API
│   ├── ai_model.py              # FinBERT wrapper
│   ├── regime_detector.py       # SMA + vol regime classifier
│   ├── backtester.py            # Quant metrics + transaction cost
│   ├── alerts.py                # Discord webhook
│   ├── experiment_tracker.py    # MLflow integration
│   └── disk_cache.py            # joblib/parquet persistence
├── ui/
│   └── components.py            # Plotly chart factory
├── tests/
│   └── test_engine.py           # 16 pytest tests
├── docs/
│   ├── index.html               # Live report (generated)
│   ├── .nojekyll                # GitHub Pages config
│   └── MEDIUM_POST.md           # Blog post draft
├── .github/
│   └── workflows/
│       ├── ci.yml               # Run tests on push/PR
│       └── rebuild-report.yml   # Weekly cron rebuild
├── Dockerfile                   # Multi-stage, non-root
├── requirements.txt
├── README.md
└── README.th.md                 # Thai translation
```

</details>

---

## 🎯 Who This Is For

**Primary user**: Retail investors with technical curiosity who want to
understand market regimes beyond fundamental ratios.

**Secondary users**:
- 🎓 ML/Quant students learning walk-forward validation, regime detection
- 💼 Junior data scientists wanting a finance-domain portfolio piece
- 📰 Anyone curious how a real "risk dashboard" is built end-to-end

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.12+ |
| **Data** | yfinance, pandas, numpy, Google News RSS |
| **ML** | scikit-learn, XGBoost, LightGBM, PyTorch (LSTM), SHAP |
| **NLP** | HuggingFace Transformers + FinBERT (ProsusAI) |
| **Viz** | Plotly (interactive + 3D + animated) |
| **App** | Streamlit |
| **Tracking** | MLflow (local file store) |
| **Alerts** | Discord webhook |
| **CI/CD** | GitHub Actions (pytest + auto-rebuild cron) |
| **Hosting** | GitHub Pages (static report), Docker, Streamlit Cloud |
| **Testing** | pytest (16 tests, all passing) |
| **Persistence** | joblib + parquet TTL-based disk cache |

---

## 📊 About the Live Report

The live report is a **pre-built static HTML snapshot** (not a live web app).

| Property | Value |
|---|---|
| 📦 Format | Self-contained HTML with embedded Plotly CDN |
| 🕐 Auto-update | Every Monday 06:00 UTC + on code push |
| ⚡ Load time | < 1 second |
| 💸 Cost | $0 (GitHub Pages) |
| 📱 Mobile | Fully responsive |

The **"📅 Generated"** timestamp at the top of the report shows last
rebuild time.

**Want real-time data?** Run `streamlit run app.py` locally.

---

## 🤝 AI-Assisted Development

Built collaboratively with **Claude (Anthropic)** as a coding assistant.
AI helped with code generation, debugging, and documentation drafts.
I owned problem definition, methodology, critical evaluation,
verification, and all design decisions.

Every line of code was reviewed, tested, and committed by me.

<details>
<summary>📖 Detailed breakdown (what AI did vs what I owned)</summary>

### What AI helped with
- 💻 Boilerplate code generation (Streamlit, Plotly chart factories, ML pipeline)
- 🐛 Debugging runtime errors + type warnings (Pylance, edge cases)
- 📐 Suggesting architecture patterns (layered separation, disk cache design)
- 📝 Drafting documentation (both READMEs, MEDIUM_POST.md)
- 🎨 HTML/CSS template scaffolding for the static report

### What I (the developer) owned
- 🎯 **Problem definition & scope** — choosing financial crisis detection as the domain
- 📊 **Methodology decisions** — walk-forward CV, naive baseline, COVID case study framing
- 🧪 **Critical evaluation** — accepting that Naive beats ML (rather than hiding it)
- ✅ **Verification** — running every test, validating every commit, debugging
- 🚀 **Deployment & iteration** — GitHub setup, Pages config, prioritization
- 📐 **Domain interpretation** — knowing what VIX > 30 means in context

### Why disclose this?
AI-assisted development is standard in 2025+ software engineering — GitHub Copilot,
Cursor, Claude Code, and similar tools are widely adopted across the industry.
Transparent disclosure is:

1. **Honest** — what you see in the repo is what was actually built
2. **Modern practice** — collaborating productively with AI is itself an engineering skill
3. **Reproducible** — anyone can fork this repo and use the same AI assistance

</details>

---

## ⚠️ Disclaimer

This is an **educational and research project**. It is **not financial advice**.
VIX is one of the hardest variables to forecast in finance — even our best
ML couldn't beat naive persistence. Use the system as a *directional signal*,
not as a trading recommendation.

Past performance does not guarantee future results. Markets can stay irrational
longer than you can stay solvent.

---

## 👤 Author

**oomNoNe** — [@oomNoNe](https://github.com/oomNoNe)

📝 [Read the blog post on Medium-style findings](docs/MEDIUM_POST.md)
🇹🇭 [README ภาษาไทย](README.th.md)

---

## 📄 License

MIT License — free to use, modify, and learn from.
