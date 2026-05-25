# 🚨 Black Swan Risk Indicator (Quant Edition)

> 🌐 **Languages:** [English](README.md) | **ภาษาไทย**

[![Live Report](https://img.shields.io/badge/📊_รายงานสด-ดูเลย-success?style=for-the-badge)](https://oomNoNe.github.io/black-swan-indicator/)

[![CI](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/oomNoNe/black-swan-indicator/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 👉 **คลิกแบดจ์เขียวด้านบน** เพื่อดูรายงานสดทันที (ไม่ต้องติดตั้งอะไร)

ระบบเตือนภัยล่วงหน้าสำหรับวิกฤตการเงิน ขับเคลื่อนด้วย AI และ Quant Analysis
ผสาน **การวิเคราะห์ Sentiment ข่าวด้วย NLP**, **การพยากรณ์ความผันผวนของตลาดด้วย ML**,
และ **การตรวจจับสภาพตลาด (Regime Detection)** เพื่อสร้าง Crisis Risk Score (0–100)
พร้อมระบบ Backtest กลยุทธ์การเทรด

---

## ✨ ฟีเจอร์หลัก

| Module | เทคโนโลยี | หน้าที่ |
|---|---|---|
| **Live Risk Dashboard** | Streamlit + Plotly | แสดง VIX แบบเรียลไทม์, วิเคราะห์ Sentiment ข่าว Google News ด้วย FinBERT, แสดงผลรวมเป็น Crisis Score Gauge |
| **AI Regime & Prediction** | XGBoost Regressor | พยากรณ์ค่า VIX ล่วงหน้า 7 วัน พร้อมแสดง Feature Importance |
| **Quant Backtest** | NumPy / Pandas | คำนวณ Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor เทียบกับ Buy & Hold |
| **Regime Detector** | SMA + Rolling Vol | จำแนกสภาพตลาดเป็น Trending Bull / Ranging / Panic แล้วปรับน้ำหนัก Risk Equation อัตโนมัติ |

---

## 🧮 The Crisis Equation (สมการความเสี่ยง)

```
Crisis_Score = w_news(regime) × News_Risk + w_market(regime) × Market_Risk
```

น้ำหนักปรับตามสภาพตลาด:

| Regime (สภาพตลาด) | w_news (น้ำหนักข่าว) | w_market (น้ำหนักตลาด) |
|---|---|---|
| 📈 Trending Bull (ขาขึ้น) | 0.3 | 0.7 |
| 🔥 Panic (วิกฤต) | 0.7 | 0.3 |
| ⚖️ Ranging / Unknown | 0.5 | 0.5 |

**เหตุผล**: ในช่วงตลาดวิกฤต ข่าวจะส่งผลต่อจิตวิทยาผู้ลงทุนมากกว่าตัวเลข
ส่วนช่วงตลาดสงบ ตัวเลขทางเทคนิคน่าเชื่อถือกว่าข่าวที่อาจมี noise สูง

**Backtest Signal** จะส่งสัญญาณเตือนเมื่อ:
**VIX > 30** ⚠️ **และ** ความผันผวน 20 วัน > 1.5 เท่าของค่ามัธยฐาน 252 วัน

---

## 🗂️ โครงสร้าง Project

```
black-swan-indicator/
├── app.py                    # Streamlit entry point (3 tabs + sidebar)
├── data/
│   ├── market_crawler.py     # ดึงข้อมูล S&P 500 + VIX จาก yfinance
│   └── news_crawler.py       # ดึงข่าวการเงินจาก Google News RSS
├── engine/
│   ├── ai_model.py           # FinBERT sentiment analyzer wrapper
│   ├── ml_predictor.py       # XGBoost VIX forecaster
│   ├── regime_detector.py    # ตรวจจับ market regime + dynamic risk weighting
│   └── backtester.py         # คำนวณ Sharpe, MDD, Win Rate, Profit Factor
├── ui/
│   └── components.py         # Plotly charts ทั้งหมด (gauge, equity curve, etc.)
└── tests/
    └── test_engine.py        # Unit tests (pytest)
```

---

## 🚀 วิธีติดตั้งและรัน

### ทางเลือก A — ติดตั้งด้วย Python (local)
```bash
git clone https://github.com/oomNoNe/black-swan-indicator.git
cd black-swan-indicator
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

### ทางเลือก B — Docker (แนะนำสำหรับ deploy)
```bash
docker build -t black-swan-indicator .
docker run -p 8501:8501 black-swan-indicator
```
เปิด http://localhost:8501

### ทางเลือก C — Streamlit Cloud (host ฟรี ไม่ต้องเซ็ตอัพ)
1. Fork repo นี้ไปยัง GitHub account ของคุณ
2. ไปที่ https://share.streamlit.io
3. คลิก **"New app"** → เลือก fork ของคุณ → branch main → `app.py`
4. กด Deploy รอ 5-10 นาที จะได้ public URL

> ⚠️ **หมายเหตุ**: ครั้งแรกที่รันจะ download โมเดล FinBERT (~440 MB)
> ใช้เวลา 2-5 นาที (ขึ้นกับความเร็วเน็ต) ครั้งต่อไปจะใช้แคชอัตโนมัติ

---

## 🧪 การพัฒนา (Development)

รัน tests:
```bash
pytest tests/ -v
```

CI จะรันอัตโนมัติทุกครั้งที่ push ผ่าน GitHub Actions ([workflow](.github/workflows/ci.yml))

---

## 🎓 ทำไมเลือกโมเดลเหล่านี้?

### 🤖 FinBERT (สำหรับ Sentiment Analysis)

**ทำไมเลือก**: ฝึกบน corpus การเงินโดยเฉพาะ — เข้าใจศัพท์ที่โมเดลทั่วไปอ่านไม่ออก
เช่น "bearish guidance", "hawkish Fed", "deleveraging"

| ✅ ข้อดี | ❌ ข้อเสีย |
|---|---|
| แม่นกว่า BERT ทั่วไป ~15% สำหรับข่าวการเงิน | รองรับแค่ภาษาอังกฤษ |
| Open-source ฟรี (ProsusAI) | ฝึกปี 2019 — ไม่รู้ศัพท์ใหม่ (GameStop, AI bubble) |
| เร็วพอใช้บน CPU | ใช้ RAM ~440MB |
| Pre-trained บน Reuters TRC2 | อาจ bias จาก training data |

**ทางเลือกที่พิจารณาแต่ไม่เลือก**:
- ❌ GPT-4 API → แพง, ต้อง API key, latency สูง
- ❌ Generic BERT → ไม่เข้าใจบริบทการเงิน
- ❌ VADER → rule-based, ไม่ใช่ ML

### 🌲 XGBoost (สำหรับพยากรณ์ VIX)

**ทำไมเลือก**: เก่งกับข้อมูลแบบตาราง (tabular) ซึ่งเป็นรูปแบบของ macro features
ของเรา (VIX lag, returns) — และชนะ Kaggle competitions มากที่สุด

| ✅ ข้อดี | ❌ ข้อเสีย |
|---|---|
| Industry-proven (Kaggle, hedge funds) | ไม่มี memory ของเวลา → ต้องสร้าง lag features เอง |
| เร็วมาก (เทรน < 1 วินาที) | Overfit ง่ายถ้าไม่ tune |
| ตีความได้ผ่าน feature importance | ไม่เก่ง extrapolation |
| ทนต่อ missing values | แพ้ deep learning บนข้อมูลใหญ่มากๆ |
| ไม่ต้อง scale features | |

**ทางเลือกที่พิจารณาแต่ไม่เลือก**:
- ❌ Linear Regression → ไม่จับ non-linearity
- ❌ LSTM → ต้องการข้อมูลเยอะกว่า, เทรนนาน, ตีความยาก
- ❌ ARIMA → assume linear & stationary (VIX ไม่ใช่)
- ❌ Transformer → overkill สำหรับ 5 features

### 📊 SMA + Rolling Vol (สำหรับ Regime Detection)

**ทำไมเลือก**: เรียบง่าย ตีความได้ทันที — ทุกกองทุนใหญ่ใช้ตัวนี้เป็นมาตรฐาน
(Bridgewater, Two Sigma) ไม่ต้องการ ML ก็ใช้งานได้

| ✅ ข้อดี | ❌ ข้อเสีย |
|---|---|
| เข้าใจง่าย ไม่ใช่ black box | **Lagging indicator** — SMA-200 ตอบสนองช้า |
| ไม่ต้อง tune hyperparameter | Binary thresholds — อาจพลาดช่วง transition |
| คำนวณเร็ว | ไม่ใช้ข้อมูล macro/news |
| Industry standard | |

**ทางเลือกที่พิจารณาแต่ไม่เลือก** (พร้อม implement ในอนาคต):
- 🔄 **Hidden Markov Model (HMM)** → จับ transition ระหว่าง regime ได้นุ่มนวลกว่า
- 🔄 **Bayesian regime switching** → ให้ probability แทน binary
- 🔄 **GMM clustering** → ไม่ต้องนิยาม regime ก่อน

---

## 📊 หลักการและ Methodology

### Time-Series Validation
- ใช้ `shuffle=False` ใน `train_test_split` เพื่อรักษาลำดับเวลา (ป้องกัน look-ahead bias)
- Train/Test split = 80/20 ตามลำดับเวลา

### Lag Features
- **VIX**: ค่า lag 1, 3, และ 7 วัน
- **S&P 500 Returns**: rolling return 1 วัน และ 5 วัน

### NLP Sentiment
- ใช้โมเดล `ProsusAI/finbert` (FinBERT) ที่ fine-tune มาเพื่อข่าวการเงินโดยเฉพาะ
- ตัดข้อความหัวข้อข่าวที่เกิน 512 tokens
- แปลงผลลัพธ์เป็นคะแนน: Negative=100, Neutral=50, Positive=0

### Caching Strategy (Streamlit)
- `@st.cache_resource` สำหรับโมเดล AI (โหลดครั้งเดียวต่อ session)
- `@st.cache_data(ttl=1h)` สำหรับข้อมูลตลาด (รีเฟรชอัตโนมัติทุก 1 ชั่วโมง)

---

## 🧪 การทดสอบ

รัน Unit Tests:
```bash
pytest tests/ -v
```

ครอบคลุม 9 test cases:
- ✅ Quant metrics calculations (Sharpe, MDD, Win Rate)
- ✅ Backtest end-to-end pipeline
- ✅ Market regime classification
- ✅ Dynamic risk equation weighting
- ✅ ML predictor training + inference
- ✅ Side-effect prevention (immutable input)

---

## 🔭 Roadmap ในอนาคต

### Tier 1 — Deploy & Polish
- [ ] Deploy ขึ้น **Streamlit Cloud** (live demo URL)
- [x] เพิ่ม **Dockerfile** (multi-stage, non-root user)
- [x] เพิ่ม **CI/CD** ด้วย GitHub Actions (รัน pytest + smoke test อัตโนมัติ)
- [x] เพิ่ม **badges** (CI status, Python version, license)
- [ ] เพิ่ม **screenshot** ใน README

### Tier 2 — ยกระดับ ML / Quant ✅ ทำเสร็จแล้ว
- [x] ใช้ **Walk-forward validation** (TimeSeriesSplit) แทน train/test split ธรรมดา
- [x] เพิ่ม **macro features**: yield curve spread, Gold, Oil, DXY (5 → 13 features)
- [x] **Classification mode**: crash vs no-crash + Precision/Recall/F1/ROC-AUC
- [x] **SHAP** feature importance (Interpretable AI)
- [x] **Transaction cost** ใน backtest (0-50 bps ปรับได้)
- [x] **Model comparison**: XGBoost vs LightGBM vs Ridge/LogReg (LSTM เลื่อนไป Tier 3)

### Tier 3 — Production-Grade ✅ 4/6 ทำเสร็จแล้ว
- [ ] Batch pipeline (Airflow) — *ข้าม: GitHub Actions cron เหมาะกับ portfolio กว่า*
- [ ] PostgreSQL store — *ข้าม: in-memory caching พอใช้ที่ scale นี้*
- [x] **Discord webhook alerts** (ตั้ง threshold + test button)
- [x] **Multi-asset** expansion (10 assets: US equity, EM, crypto, commodities)
- [x] **LSTM** model เพิ่มใน comparison (PyTorch)
- [x] **MLflow** experiment tracking (local file store)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.12+ |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Data Source** | yfinance, Google News RSS |
| **Machine Learning** | XGBoost, scikit-learn |
| **NLP** | Hugging Face Transformers (FinBERT), PyTorch |
| **Visualization** | Plotly |
| **Testing** | pytest |
| **Version Control** | Git + GitHub |

---

## 📚 ที่มาของชื่อ "Black Swan"

แนวคิด **Black Swan Event** มาจากหนังสือของ **Nassim Nicholas Taleb** หมายถึงเหตุการณ์ที่:

1. **ไม่คาดคิด** (Outlier) — อยู่นอกขอบเขตของความคาดหวังปกติ
2. **ส่งผลกระทบรุนแรง** (Extreme Impact)
3. **อธิบายได้หลังเกิดเหตุ** (Retrospectively Predictable)

ตัวอย่าง: Financial Crisis 2008, COVID-19 Crash 2020, Dot-com Bust 2000

ระบบนี้พยายาม**ตรวจจับสัญญาณก่อนเกิดเหตุ** จากตัวชี้วัดทั้งเชิงปริมาณ (VIX, vol)
และเชิงคุณภาพ (sentiment ข่าว) — แม้จะไม่สามารถทำนายได้ 100%
แต่ช่วยลดความเสียหายได้หากระบบ trigger สัญญาณทันการณ์

---

## ⚠️ คำเตือน (Disclaimer)

โปรเจกต์นี้สร้างขึ้นเพื่อ **การศึกษาและวิจัย** เท่านั้น
**ไม่ใช่คำแนะนำในการลงทุน** การตัดสินใจลงทุนใดๆ ควรพิจารณาจาก
แหล่งข้อมูลที่เชื่อถือได้และที่ปรึกษาทางการเงินมืออาชีพ

VIX และตลาดการเงินมีพฤติกรรมที่คาดเดายากมาก แม้แต่โมเดล ML ที่ดีที่สุด
ก็มักให้ค่า R² ใกล้ศูนย์ ผู้ใช้ควรใช้ผลลัพธ์เป็น *สัญญาณเชิงทิศทาง*
ไม่ใช่ความจริงสัมบูรณ์

---

## 👤 ผู้พัฒนา

**oomNoNe**
GitHub: [@oomNoNe](https://github.com/oomNoNe)

---

## 📄 License

MIT License — ใช้งานได้อย่างอิสระ
