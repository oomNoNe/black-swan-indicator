# 🚨 Black Swan Risk Indicator

> 🌐 **Languages:** [English](README.md) | **ภาษาไทย**

[![Live Report (TH)](https://img.shields.io/badge/📊_รายงานสด-ภาษาไทย-success?style=for-the-badge)](https://oomNoNe.github.io/black-swan-indicator/)
[![Live Report (EN)](https://img.shields.io/badge/📊_Live_Report-English-blue?style=for-the-badge)](https://oomNoNe.github.io/black-swan-indicator/index.en.html)

[![CI](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## นี่คืออะไร?

**Black Swan Risk Indicator** คือระบบเตือนภัยล่วงหน้าก่อนวิกฤตการเงิน
ที่ผสาน **NLP sentiment** วิเคราะห์ข่าวการเงินทั่วโลก,
**ML forecasting** แบบ time-series, และ **rule-based regime detection**
เพื่อสร้าง **Crisis Risk Score (0–100)** ที่ผ่านการ backtest กับข้อมูลย้อนหลัง 5 ปี

ระบบนี้ช่วยตรวจจับช่วงที่ตลาดกำลังจะเข้าสู่ "fear regime"
เพื่อให้นักลงทุนเตรียมตัวรับสถานการณ์ตลาดตก เช่น Subprime 2008,
COVID-2020 หรือ เงินเฟ้อ 2022

🌐 **[ดู Live Report →](https://oomNoNe.github.io/black-swan-indicator/)** · สร้างด้วย Python · MIT License

---

## 🔥 ปัญหาที่แก้

นักลงทุนรายย่อยส่วนใหญ่ **โดนวิกฤตเล่นงานแบบไม่ทันตั้งตัว** เพราะเครื่องมือที่ใช้
ส่วนใหญ่ **มองย้อนหลัง** ไม่ใช่มองข้างหน้า:

- 📊 **Stock screeners** บอก fundamentals — ใช้ไม่ได้ในช่วงตลาด panic
- 📈 **Technical indicators** ตอบสนองช้า — ยืนยัน crash หลังจาก crash เริ่มแล้ว
- 📰 **News feeds** มาทีหลัง — กว่าข่าวจะออก ตลาดก็ตกไปไกลแล้ว
- 🤖 **AI tools** ส่วนใหญ่ทำนาย "ผลตอบแทน" — แต่จริงๆ **ความผันผวน** ต่างหากที่ทำลายพอร์ต

ขณะเดียวกัน **quant มืออาชีพ** ใช้ ensemble ของ macro indicators + sentiment +
regime models ซึ่งคนทั่วไปเข้าไม่ถึง

**โปรเจกต์นี้ถาม**: *สร้าง crisis detector ที่ open-source โปร่งใส
ใช้บน laptop คนทั่วไปได้ — เป็นไปได้มั้ย?*

---

## 💡 ทางแก้

Pipeline ตรวจจับ **3 สัญญาณอิสระ** หลอมรวมกัน:

1. 📰 **News sentiment** (FinBERT วิเคราะห์พาดหัวข่าวการเงินทั่วโลก)
2. 📊 **Market volatility** (VIX + realized vol 20 วัน + threshold rules)
3. 🤖 **ML forecasting** (XGBoost บน 13 features ทำนาย VIX 7 วันข้างหน้า)

ทั้ง 3 ป้อนเข้า **Crisis Equation** ที่ปรับน้ำหนักตาม **regime ตลาดปัจจุบัน**
(Bull / Panic / Ranging)

เมื่อ score เกิน **70** ระบบจะ flag วันนั้นเป็น high-risk บน dashboard
เพื่อให้คุณตอบสนองก่อนข่าวจะออก

---

## 🗺️ How It Works (User Journey)

```
Daily / On-demand:

┌─────────────────────────────────────────────────────────────┐
│  1. Fetch data                                              │
│     yfinance: S&P 500, VIX, Treasury 10Y/3M, Gold, Oil,     │
│     DXY (~1,200 days, 8 series)                             │
│     Google News RSS: 10 latest financial headlines          │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Feature engineering                                     │
│     13 features: VIX lag (1/3/7 days), momentum,            │
│     S&P returns (1d/5d), realized vol, yield curve          │
│     spread, inversion flag, Gold/Oil/DXY momentum           │
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
│     • Backtest equity + drawdown vs Buy & Hold              │
│     • Multi-asset volatility comparison (10 assets)         │
│     • COVID-19 case study & ML vs Naive baseline finding    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Presentation Layer                                              │
│  ├── Streamlit app (5 tabs + sliders)                            │
│  └── Static HTML report (rebuilt weekly via cron)                │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Engine Layer (engine/)                                          │
│  ├── features.py            — 13-feature builder                 │
│  ├── ml_predictor.py        — 5 models (XGB, LGBM, Ridge,        │
│  │                             LSTM, Naive baseline)             │
│  ├── ai_model.py            — FinBERT wrapper                    │
│  ├── regime_detector.py     — Market mood classifier             │
│  ├── backtester.py          — Sharpe / MDD / transaction cost    │
│  └── disk_cache.py          — joblib + parquet persistence       │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Data Layer (data/)                                              │
│  ├── market_crawler.py      — yfinance (10 assets)               │
│  └── news_crawler.py        — Google News RSS                    │
└────────────────────────────────┬─────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────┐
│  Infrastructure                                                  │
│  ├── GitHub Actions         — CI/CD + weekly cron rebuild        │
│  ├── GitHub Pages           — Free static report hosting         │
│  ├── Docker                 — Containerized deployment           │
│  └── Streamlit Cloud        — Optional managed hosting           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Findings (ผลที่ honest ไม่หมก)

โปรเจกต์นี้เน้น **intellectual honesty มากกว่า hype**

### 🏆 Finding 1: Naive baseline ชนะทุก ML model
หลัง tune 4 โมเดล (XGBoost, LightGBM, Ridge, LSTM) ด้วย walk-forward validation
**persistence baseline** (`VIX อีก 7 วัน = VIX วันนี้`) กลับชนะทั้งหมด:

| Model | Mean R² (walk-forward CV, 5 folds) |
|---|---|
| 🥇 **Naive (baseline)** | **+0.093** |
| 🥈 XGBoost (tuned, regularized) | −0.122 |
| 🥉 LightGBM | −0.184 |
| Ridge | −0.131 |
| LSTM (PyTorch) | −3.720 |

**แปลความ**: VIX มีพฤติกรรม random walk สูงมาก ยืนยันทฤษฎี
**Efficient Market Hypothesis** ของ Fama บนชั้น volatility-of-volatility

🎓 **บทเรียน Engineering**: ทดสอบ naive baseline ก่อนเสมอ
ถ้า ML แฟนซีไม่ชนะ persistence — **อย่า ship**

### 🦠 Finding 2: Rule-based ทำงานได้ในวิกฤต COVID-2020
แม้ ML forecasting ไม่แม่น แต่ rule-based detector
(`VIX > 30 AND vol_spike > 1.5×`) จับ COVID crash ได้
ในวันที่ **9 มี.ค. 2020** — **7 วันก่อน** VIX แตะจุดสูงสุด 82.69 (16 มี.ค.)

นักลงทุนที่ฟังสัญญาณ → หลีกเลี่ยงการขาดทุน ~30%

### 💰 Finding 3: Transaction costs สำคัญมาก
ถ้าไม่คิดค่าธรรมเนียม กลยุทธ์ดูดีกว่าจริง พอใส่ 10 bps ต่อ turnover
Sharpe ลด ~0.05 (น้อยแต่มีผล) **paper academic หลายฉบับข้ามตรงนี้**
ทำให้กลยุทธ์ดูดีเกินจริง

---

## 🎓 Research Methodology (วิธีวิจัย)

ออกแบบเป็น **comparative study** ตอบคำถาม: *การใช้ news sentiment + ML
forecasting + rule-based regime detection ให้ risk-adjusted return ดีกว่า
Buy & Hold หรือไม่?*

| ขั้นตอน | สิ่งที่กำหนด |
|---|---|
| **H₀ / H₁** | `Sharpe_strategy ≤ Sharpe_BH` vs `Sharpe_strategy > Sharpe_BH` |
| **α** | 0.05 · walk-forward k=5 · ML ทุกตัวต้องชนะ Naive baseline |
| **Test stats** | R², Sharpe Δ, Max Drawdown, Win Rate, Profit Factor, COVID lead-time |
| **ข้อมูล** | 1,259 วันทำการ · 8 macro series · 10 bps transaction cost |
| **สรุปผล** | ❌ ML แพ้ Naive · ✅ Rule-based จับ COVID ก่อน peak 7 วัน · ⚠️ Defensive แพ้ B&H ในตลาด bull |

<details>
<summary>📖 รายละเอียดเต็ม 5 ขั้นตอน (คลิกเปิด)</summary>

### 1. ตั้งสมมติฐาน (Hypothesis)
- **H₀ (Null)**: กลยุทธ์ Black Swan ไม่ดีกว่า Buy & Hold ในแง่ risk-adjusted
  `Sharpe_strategy ≤ Sharpe_BH`
- **H₁ (Alternative)**: กลยุทธ์ Black Swan ให้ Sharpe ดีกว่า
  `Sharpe_strategy > Sharpe_BH`

### 2. กำหนดลำดับนัยสำคัญ (Significance Level)
- **α = 0.05** สำหรับ formal tests (มาตรฐาน academic)
- ใช้ walk-forward CV **k = 5 folds** ลด bias จาก single train/test split
- ทุก ML forecast ต้องเทียบกับ **Naive persistence baseline**
  (ถ้าไม่ชนะ baseline = ไม่ควร deploy)

### 3. เลือกสถิติทดสอบ (Test Statistic)
| คำถาม | Metric ที่ใช้ |
|---|---|
| ความแม่นของ forecast | **R² (walk-forward mean ± std)** เทียบ Naive |
| Strategy ดีกว่ามั้ย | **Sharpe Ratio** delta (strategy − B&H) |
| ลด tail risk ได้มั้ย | **Max Drawdown** เปรียบเทียบ |
| คุณภาพการเทรด | **Win Rate** + **Profit Factor** |
| ตรวจจับวิกฤต | Lead time ก่อน peak (case study COVID) |

### 4. คำนวณ (Calculation)
- **ข้อมูล**: ~1,259 วันทำการ (5 ปี), 8 macro series จาก yfinance
- **Realism**: transaction cost = 10 bps ต่อ turnover
- **Validation**: TimeSeriesSplit (ไม่ shuffle, ป้องกัน look-ahead bias)
- **Reproducible**: cached models ใน `.cache/` สำหรับ replication

### 5. สรุปผล (Conclusion)
- ❌ **ML ทำนายไม่ชนะ Naive baseline** (R² ≈ 0.09 สำหรับ Naive,
  ติดลบทุก ML model) — **ไม่สามารถ reject H₀** ในเชิง forecasting
- ✅ **Rule-based crisis detector ผ่านการทดสอบใน COVID-19**
  (สัญญาณเตือนก่อน VIX peak 7 วัน)
- ⚠️ **ในตลาด bull, กลยุทธ์ defensive แพ้ Buy & Hold** —
  คุณค่าของ crash-avoidance ปรากฏเฉพาะช่วงวิกฤตจริง

**Future statistical work**: Diebold-Mariano test สำหรับ forecast accuracy,
bootstrap CI สำหรับ Sharpe difference, paired t-test บน daily returns

</details>

---

## 🧠 Engineering Challenges

5 trade-offs ที่ไม่ obvious ระหว่างพัฒนา:

| # | Challenge | สรุปสั้น |
|---|---|---|
| 1 | Walk-forward vs naive split | เลือก CV ช้ากว่า 5-10× แต่ honest |
| 2 | LSTM บน data เล็ก | บันทึก R² = -3.72 ตรงๆ เก็บไว้แสดง breadth |
| 3 | Static vs Streamlit | Hybrid: HTML แชร์ + Streamlit dev |
| 4 | Rebuild ช้า | Disk cache 7× speedup (60s → 8.7s) |
| 5 | FinBERT vocab drift | ยอมรับข้อจำกัด vs จ่าย GPT-4 API |

<details>
<summary>📖 อ่านรายละเอียดแต่ละข้อ (พร้อม alternatives ที่พิจารณา)</summary>

### 1. เลือก validation strategy
**ปัญหา**: `train_test_split` แบบ random → leak อนาคต → R² สวยปลอม

**แก้**: เปลี่ยนเป็น `TimeSeriesSplit` แบบ **walk-forward expanding window**
Train [0..t], test [t..t+k], rolling forward เหมือนใช้งานจริง

**Trade-off**: ช้ากว่า 5-10× แต่เป็นวิธีเดียวที่ honest

### 2. Dataset เล็กเกินสำหรับ Deep Learning
**ปัญหา**: ~1,200 วันน้อยเกินสำหรับ LSTM → overfit หนัก (R² −3.72)

**แก้**: ระบุข้อจำกัดใน report. เก็บ LSTM ไว้ใน comparison เพื่อแสดง breadth
แต่แนะนำใช้ Ridge/Naive ใน production. **บทเรียน**: deep learning ไม่ใช่
คำตอบเสมอ

### 3. Trade-off static vs live Streamlit
**ปัญหา**: Streamlit Cloud free (1 GB RAM) host FinBERT (440 MB) +
XGBoost + multi-asset ไม่ไหว Cold start นาน 5+ นาที

**แก้**: Hybrid architecture
- **Static HTML** (rebuild ทุกสัปดาห์) → แชร์ใครก็เปิดได้ทันที
- **Streamlit local** → สำหรับ deep-dive แบบ interactive

Static report เสิร์ฟ 99% ของ viewer ใน <1s. Streamlit อยู่สำหรับ
1% ที่ต้องการ tune parameter

### 4. Persistent disk cache สำหรับ rebuild
**ปัญหา**: รัน `build_report.py` แต่ละครั้งใช้ ~60s (fetch + train +
walk-forward × 4 model + SHAP)

**แก้**: สร้าง `engine/disk_cache.py` — TTL-based parquet + joblib cache
รัน 2nd time ลดเหลือ **~8.7s (7× เร็วขึ้น)**

### 5. Sentiment drift ของ FinBERT
**ปัญหา**: FinBERT trained ปี 2019 ไม่รู้จักศัพท์ใหม่ ("AI bubble",
"GameStop saga")

**Trade-off ยอมรับ**: ระบุข้อจำกัดใน glossary. ทางเลือก (GPT-4 API)
แพง + latency สูง สำหรับ open-source educational project — FinBERT
ยังเป็นตัวเลือกที่เหมาะ

</details>

---

## 🚀 Quick Start

### Option A — แค่เปิด live report
👉 **[oomNoNe.github.io/black-swan-indicator/](https://oomNoNe.github.io/black-swan-indicator/)** (ไม่ต้องติดตั้งอะไร)

### Option B — ติดตั้งเอง (local Python)
```bash
git clone https://github.com/oomNoNe/black-swan-indicator.git
cd black-swan-indicator
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# สร้าง static report (8.7s หลัง first run, ~60s ครั้งแรก)
python scripts/build_report.py
start docs/index.html

# หรือรัน Streamlit app แบบ interactive
streamlit run app.py
```

### Option C — Docker
```bash
docker build -t black-swan-indicator .
docker run -p 8501:8501 black-swan-indicator
```

---

## 🎯 ใครคือ User?

**Primary user**: นักลงทุนรายย่อยที่มี technical curiosity
อยากเข้าใจ market regime ลึกกว่าดูแค่ fundamental

**Secondary users**:
- 🎓 นักศึกษาสาย ML/Quant ที่เรียน walk-forward, regime detection
- 💼 Junior data scientist ที่อยากได้ portfolio piece สาย finance
- 📰 ใครที่สงสัยว่า "risk dashboard" จริงๆ สร้างกันยังไง

---

## 📚 Data Sources & References (แหล่งข้อมูลและอ้างอิง)

### แหล่งข้อมูล

| Source | ใช้ทำอะไร | การเข้าถึง | หมายเหตุ |
|---|---|---|---|
| **Yahoo Finance** (ผ่าน [yfinance](https://github.com/ranaroussi/yfinance)) | S&P 500, VIX, 10Y/3M Treasury, Gold, Oil, DXY, multi-asset | ฟรี (unofficial scraper) | Daily OHLCV, delay ~15 นาที, มี rate limit |
| **Google News RSS** | พาดหัวข่าวการเงิน (เฉพาะ title) | RSS feed ฟรี | ภาษาอังกฤษเท่านั้น, ~10 ข่าว/query, ไม่มี historical |
| **Hugging Face Hub** | โมเดล `ProsusAI/finbert` pre-trained | ฟรี (ต้องสมัคร HF) | ~440 MB ดาวน์โหลดครั้งแรก, cache อัตโนมัติ |

### Ticker Symbols ที่ใช้ (yfinance)

```
^GSPC    — S&P 500 Index           ^VIX    — CBOE Volatility Index
^TNX     — 10-Year Treasury Yield  ^IRX    — 13-Week T-Bill Yield
GC=F     — Gold Futures            CL=F    — WTI Crude Oil Futures
DX-Y.NYB — US Dollar Index (DXY)
^NDX     — Nasdaq 100              ^VXN    — Nasdaq Volatility
^RUT     — Russell 2000            ^RVX    — Russell Volatility
^OVX     — Oil Volatility Index
EEM      — Emerging Markets ETF    FXI     — China Large-Cap ETF
EWZ      — Brazil ETF              BTC-USD — Bitcoin / USD
ETH-USD  — Ethereum / USD
```

### Academic References (งานวิจัยอ้างอิง)

Methodology ของ project นี้ต่อยอดจากงานวิจัยมาตรฐาน:

- **Random Walk Hypothesis** — Fama, E. (1970). *Efficient Capital Markets:
  A Review of Theory and Empirical Work*. Journal of Finance, 25(2), 383–417.
  ผู้ได้รับ Nobel เศรษฐศาสตร์ปี 2013
- **XGBoost** — Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree
  Boosting System*. KDD '16
- **LightGBM** — Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient
  Boosting Decision Tree*. NIPS '17
- **SHAP values** — Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to
  Interpreting Model Predictions*. NIPS '17 (ต่อยอดจาก Shapley 1953, Nobel)
- **FinBERT** — Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with
  Pre-trained Language Models*. arXiv:1908.10063
- **Walk-Forward Validation** — Bergmeir, C. & Benítez, J. M. (2012). *On the
  use of cross-validation for time series predictor evaluation*. Information Sciences, 191, 192–213

### License & ข้อจำกัด

- **yfinance** เป็น open-source ที่ scrape Yahoo Finance — Yahoo terms of
  service applies, ห้าม commercial redistribution ของ raw data
- **Google News** RSS เปิดสาธารณะ ใช้งานตาม terms ของ Google
- **FinBERT** ออกใต้ MIT-like license บน Hugging Face
- **ข้อมูล cache** ทั้งหมดใน `.cache/` ถูก gitignore — ephemeral
- **ไม่การันตี data quality** — yfinance อาจมี missing bars, delay,
  หรือ rate-limit error บางครั้ง สำหรับ production แนะนำใช้ paid provider
  เช่น Polygon.io, Refinitiv, Bloomberg

---

## 📊 เกี่ยวกับ Live Report

Live report เป็น **static HTML snapshot** ที่สร้างไว้ล่วงหน้า (ไม่ใช่ live web app)

| Property | Value |
|---|---|
| 📦 Format | Self-contained HTML + Plotly CDN |
| 🕐 Auto-update | ทุกวันจันทร์ 06:00 UTC + ตอน push code |
| ⚡ โหลด | < 1 วินาที |
| 💸 ค่าใช้จ่าย | $0 (GitHub Pages) |
| 📱 Mobile | รองรับเต็ม |

**"📅 อัพเดทล่าสุด"** ด้านบน report บอกเวลา rebuild ล่าสุด

**อยากได้ real-time?** รัน `streamlit run app.py` ในเครื่อง

---

## 🤝 AI-Assisted Development

พัฒนาร่วมกับ **Claude (Anthropic)** เป็น coding assistant
AI ช่วยเรื่อง code generation, debugging, และร่าง documentation
ผมเป็นคนตัดสินใจ problem definition, methodology, critical evaluation,
verification, และ design ทั้งหมด

โค้ดทุกบรรทัดผ่านการ review, test, และ commit โดยตัวผมเอง

<details>
<summary>📖 รายละเอียด (AI ช่วยอะไร vs ผมทำอะไร)</summary>

### สิ่งที่ AI ช่วย
- 💻 Boilerplate code (Streamlit, Plotly charts, ML pipeline)
- 🐛 Debug runtime errors + type warnings (Pylance, edge cases)
- 📐 แนะนำ architecture patterns (layered separation, disk cache design)
- 📝 ร่าง documentation (READMEs ทั้ง 2 ภาษา, MEDIUM_POST.md)
- 🎨 HTML/CSS template scaffolding สำหรับ static report

### สิ่งที่ผม (developer) เป็นคนตัดสินใจ
- 🎯 **Problem definition + scope** — เลือก domain "financial crisis detection"
- 📊 **Methodology decisions** — walk-forward CV, naive baseline, COVID case study
- 🧪 **Critical evaluation** — ยอมรับว่า Naive ชนะ ML (แทนที่จะปกปิด)
- ✅ **Verification** — รัน tests ทุกรอบ, validate ผลทุก commit, debug เอง
- 🚀 **Deployment + iteration** — GitHub setup, Pages config, การจัด priority
- 📐 **Domain interpretation** — รู้ว่า VIX > 30 หมายถึงอะไรในบริบทจริง

### ทำไมต้องเปิดเผย?
AI-assisted development เป็นเรื่องปกติใน software engineering ปี 2025+
GitHub Copilot, Cursor, Claude Code และเครื่องมือใกล้เคียง
ใช้กันทั่วอุตสาหกรรม การเปิดเผยตรงๆ:

1. **Honest** — สิ่งที่เห็นใน repo คือสิ่งที่ทำจริง
2. **Modern practice** — การใช้ AI ให้ productive คือ engineering skill ของยุคนี้
3. **Reproducible** — ใครก็ fork repo นี้ใช้ AI assistance เดียวกันได้

</details>

---

## ⚠️ คำเตือน

โปรเจกต์นี้สร้างเพื่อ **การศึกษาและวิจัย** เท่านั้น **ไม่ใช่คำแนะนำการลงทุน**
VIX เป็นหนึ่งใน variable ที่พยากรณ์ยากที่สุดในการเงิน — แม้ ML ที่ดีที่สุด
ของเราก็ยังแพ้ naive persistence ใช้ระบบเป็น *สัญญาณทิศทาง* เท่านั้น
ไม่ใช่คำแนะนำเทรด

ผลในอดีตไม่การันตีอนาคต ตลาดอาจ irrational ได้นานกว่าที่คุณ solvent อยู่

---

## 👤 Author

**oomNoNe** — [@oomNoNe](https://github.com/oomNoNe)

📝 [อ่าน Medium-style blog post](docs/MEDIUM_POST.md)
🌐 [English README](README.md)

---

## 📄 License

MIT License — ใช้งาน, ดัดแปลง, เรียนรู้ได้อย่างอิสระ
