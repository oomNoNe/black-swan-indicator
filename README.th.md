# 🚨 Black Swan Risk Indicator

> 🌐 **Languages:** [English](README.md) | **ภาษาไทย**

[![Live Report](https://img.shields.io/badge/📊_รายงานสด-ดูเลย-success?style=for-the-badge)](https://oomNoNe.github.io/black-swan-indicator/)

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

เมื่อ score เกิน **70** ระบบส่ง **Discord webhook alert** ได้ทันที
เพื่อให้คุณตอบสนองก่อนข่าวจะออก

---

## 🗺️ How It Works (User Journey)

```
รายวัน / On-demand:

┌─────────────────────────────────────────────────────────┐
│  1. ดึงข้อมูล                                            │
│     yfinance: S&P 500, VIX, Treasury 10Y/3M, Gold, Oil, │
│     DXY (~1,200 วัน, 8 series)                          │
│     Google News RSS: 10 พาดหัวข่าวการเงินล่าสุด        │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  2. Feature engineering                                  │
│     13 features: VIX lag (1/3/7 วัน), momentum,         │
│     S&P returns (1d/5d), realized vol, yield curve      │
│     spread, inversion flag, Gold/Oil/DXY momentum       │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  3. 3 pipelines ทำงานคู่ขนาน                            │
│     ┌─────────────┐ ┌──────────────┐ ┌────────────────┐│
│     │ FinBERT     │ │ XGBoost      │ │ Regime         ││
│     │ sentiment   │ │ ทำนาย VIX    │ │ classifier     ││
│     │ (NEG/NEU/   │ │ 7 วัน +      │ │ (SMA-50/200    ││
│     │  POS)       │ │ walk-forward │ │  + rolling vol)││
│     └─────────────┘ └──────────────┘ └────────────────┘│
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  4. Crisis Equation (รวมแบบ regime-weighted)            │
│     Score = w_news(regime) × NewsRisk                   │
│           + w_market(regime) × MarketRisk                │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  5. Outputs                                              │
│     • Crisis Risk Score (0–100) บน gauge                │
│     • Backtest equity + drawdown vs Buy & Hold          │
│     • เปรียบเทียบ volatility ข้าม asset (10 assets)     │
│     • Discord alert ถ้า Score เกิน threshold            │
└─────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Presentation Layer                                       │
│  ├── Streamlit app (5 tabs + sliders)                    │
│  └── Static HTML report (rebuilt อัตโนมัติทุกสัปดาห์)    │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Engine Layer (engine/)                                   │
│  ├── features.py        — 13-feature builder              │
│  ├── ml_predictor.py    — 5 โมเดล (XGB, LGBM, Ridge,     │
│  │                         LSTM, Naive baseline)          │
│  ├── ai_model.py        — FinBERT wrapper                 │
│  ├── regime_detector.py — Market mood classifier          │
│  ├── backtester.py      — Sharpe / MDD / transaction cost│
│  ├── alerts.py          — Discord webhook                 │
│  ├── experiment_tracker.py — MLflow                       │
│  └── disk_cache.py      — joblib + parquet persistence    │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Data Layer (data/)                                       │
│  ├── market_crawler.py  — yfinance (10 assets)           │
│  └── news_crawler.py    — Google News RSS                 │
└──────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────┐
│  Infrastructure                                            │
│  ├── GitHub Actions      — CI/CD + weekly cron rebuild    │
│  ├── GitHub Pages        — Host รายงาน static ฟรี         │
│  ├── Docker              — Containerized deployment       │
│  └── Streamlit Cloud     — Optional managed hosting       │
└──────────────────────────────────────────────────────────┘
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

## 🧠 Engineering Challenges

การตัดสินใจที่ไม่ obvious และใช้เวลานานสุด:

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

## 🔭 Roadmap (Scoped + Honest)

### Tier 1: ทำเสร็จแล้ว ✅
- [x] Walk-forward validation + 5-model comparison
- [x] FinBERT sentiment + Crisis Equation
- [x] Static HTML report + GitHub Pages auto-deploy
- [x] Persistent disk cache (joblib + parquet)
- [x] Docker + CI/CD + 16 unit tests
- [x] COVID-2020 case study
- [x] Multi-asset (10 assets)
- [x] Discord webhook alerts

### Tier 2: ทำต่อไป
- [ ] Intraday data ผ่าน Polygon.io (ปัจจุบัน daily)
- [ ] Sentiment จาก Reuters/Bloomberg RSS (signal ดีกว่า Google News)
- [ ] Hidden Markov Model regime (smoother กว่า SMA threshold)
- [ ] เพิ่ม credit spread (HY-IG OAS) เป็น feature
- [ ] Out-of-sample test บน data วิกฤต 2008

### Tier 3: Research directions
- [ ] Reinforcement learning สำหรับ position sizing
- [ ] Transformer-based sentiment (เปลี่ยน FinBERT → FinGPT)
- [ ] Bayesian uncertainty quantification

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
