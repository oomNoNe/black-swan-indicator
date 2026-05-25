# 🚨 Black Swan Risk Indicator (Quant Edition)

> 🌐 **Languages:** [English](README.md) | **ภาษาไทย**

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

### 1. Clone Repository
```bash
git clone https://github.com/oomNoNe/black-swan-indicator.git
cd black-swan-indicator
```

### 2. สร้าง Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 4. รัน App
```bash
streamlit run app.py
```

จากนั้นเปิด browser ไปที่ **http://localhost:8501**

> ⚠️ **หมายเหตุ**: ครั้งแรกที่รันจะ download โมเดล FinBERT (~440 MB)
> ใช้เวลา 2-5 นาที (ขึ้นกับความเร็วเน็ต) ครั้งต่อไปจะใช้แคชอัตโนมัติ

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
- [ ] เพิ่ม **Dockerfile** สำหรับ containerization
- [ ] เพิ่ม **CI/CD** (GitHub Actions รัน pytest อัตโนมัติ)
- [ ] เพิ่ม **screenshot** ใน README

### Tier 2 — ยกระดับ ML / Quant
- [ ] ใช้ **Walk-forward validation** แทน single train/test split
- [ ] เพิ่ม features ใหม่:
  - Yield curve inversion (10Y - 2Y spread)
  - Credit spread (HY - IG)
  - Gold, Oil, USD Index
  - Put/Call ratio
- [ ] เปลี่ยนเป็น **Classification problem** (crash vs no-crash) → ใช้ Precision/Recall/F1
- [ ] เพิ่ม **SHAP** อธิบายการตัดสินใจของโมเดล (Interpretable AI)
- [ ] เพิ่ม **Transaction cost** (~0.05-0.10% per turnover) ในการ backtest

### Tier 3 — Production-Grade
- [ ] Batch pipeline ด้วย **Airflow / Prefect** + เก็บข้อมูลลง **PostgreSQL / DuckDB**
- [ ] ระบบ **Alert** ผ่าน Line Notify / Discord เมื่อ Crisis Score > 70
- [ ] ขยายขอบเขตจาก US เป็น **Multi-asset** (Crypto, EM Equities, Commodities)
- [ ] เทียบหลายโมเดล: XGBoost vs **LightGBM** vs **LSTM** vs **Transformer**
- [ ] เพิ่ม **MLflow** สำหรับ experiment tracking

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
