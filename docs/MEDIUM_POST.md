# 🚨 ผมพยายามทำนายวิกฤตการเงินด้วย XGBoost — แล้วได้เรียนรู้ว่า "การเดาแบบไม่ใช้ AI" ดีกว่า

> *Building a Black Swan Risk Indicator: ทำไม Naive baseline ชนะ Deep Learning บน VIX*

**TL;DR**: ผมสร้างระบบเตือนภัยล่วงหน้าสำหรับวิกฤตการเงินด้วย Python — รวม XGBoost, LightGBM,
LSTM, FinBERT, SHAP, MLflow ครบทุกอย่างที่ ML practitioner ควรรู้
แต่กลับพบว่า **"persistence baseline" (เดาว่า VIX พรุ่งนี้ = วันนี้) ชนะทุกโมเดล**
ที่ผมเทรน นี่คือบทเรียนสำคัญที่อยากแชร์

🔗 **Live demo**: https://oomNoNe.github.io/black-swan-indicator/
🔗 **GitHub**: https://github.com/oomNoNe/black-swan-indicator

---

## 🎯 จุดเริ่มต้น: ทำไมต้องทำ Black Swan Detector?

**Black Swan** = เหตุการณ์ที่ไม่คาดคิด ส่งผลกระทบรุนแรง อธิบายได้หลังเกิด
(Nassim Taleb, 2007) ตัวอย่าง: วิกฤต Subprime 2008, COVID-2020, สงครามรัสเซีย-ยูเครน

ทุกครั้งที่เกิดวิกฤต **VIX** (Volatility Index) จะพุ่งจาก 15-20 ไปสูงกว่า 40
ถ้าระบบจับสัญญาณได้ก่อน → ลดความเสียหายได้

**คำถาม**: เราใช้ ML ทำนาย VIX 7 วันข้างหน้าได้มั้ย?

---

## 🛠️ Stack ที่ใช้

```
Data Layer
├── yfinance — ดึง S&P 500, VIX, Treasury yields, Gold, Oil, DXY
└── Google News RSS — ดึงพาดหัวข่าวการเงิน

Engine
├── FinBERT (ProsusAI) — sentiment analysis บนข่าวการเงิน
├── XGBoost / LightGBM / Ridge / LSTM — predict VIX
├── SHAP — explainable AI
└── MLflow — experiment tracking

UI
└── Streamlit + Plotly (interactive) + static HTML report
```

---

## 📊 Feature Engineering — สร้าง 13 features

```python
# Lag features (อดีตช่วยทำนายปัจจุบัน)
'VIX_Lag1', 'VIX_Lag3', 'VIX_Lag7'
'VIX_Change_5D', 'VIX_MA_Ratio'

# S&P 500
'SP500_Return_1D', 'SP500_Return_5D', 'SP500_Vol_20D'

# Macro
'YC_Spread'   # 10Y - 3M (inverted = recession warning)
'YC_Inverted' # binary flag
'Gold_Return_5D'  # safe haven
'Oil_Return_5D'   # inflation proxy
'DXY_Return_5D'   # flight to USD
```

---

## 🧪 Walk-Forward Validation (ไม่ใช่ train/test split ธรรมดา!)

ปัญหา: ถ้า shuffle ข้อมูล time-series แล้ว split — โมเดลเห็นอนาคต → R² สูงปลอม

แก้ด้วย `TimeSeriesSplit`:
```
Fold 1: train [0..200],   test [200..400]
Fold 2: train [0..400],   test [400..600]
Fold 3: train [0..600],   test [600..800]
Fold 4: train [0..800],   test [800..1000]
Fold 5: train [0..1000],  test [1000..1200]
```

---

## 💥 Plot Twist: Naive Baseline ชนะทุก ML model

ผลลัพธ์จาก 5 folds:

| Model | Mean R² | Std |
|---|---|---|
| 🥇 **Naive** ("VIX 7d = VIX วันนี้") | **0.093** | 0.234 |
| 🥈 XGBoost (tuned) | -0.122 | 0.542 |
| 🥉 LightGBM (tuned) | -0.184 | 0.623 |
| Ridge | -0.131 | 0.296 |
| LSTM (PyTorch) | -3.720 | 7.057 |

**Wait what?** 🤯

หลังจากเทรน XGBoost 100 trees, LightGBM tuned, Ridge regression, แม้แต่ LSTM —
**ไม่มีตัวไหนชนะการเดาง่ายๆ ว่า "พรุ่งนี้ = วันนี้"**

---

## 🎓 ทำไมถึงเป็นแบบนี้? — Random Walk Hypothesis

ใน finance มีทฤษฎีหนึ่งที่ Eugene Fama (Nobel Laureate 2013) เสนอ:

> "Stock prices follow a random walk"

VIX ก็เหมือนกัน — มี **deterministic component น้อยมาก** เทียบกับ **noise**
ดังนั้นโมเดลที่ดีที่สุดคือ "ค่าวันนี้" (เพราะค่าพรุ่งนี้จะใกล้เคียง + random shock)

นี่เรียกว่า **persistence model** เป็น baseline มาตรฐานใน weather forecasting,
energy demand, และ macro economics

**บทเรียน**: ก่อนใช้ ML ใหญ่ๆ ให้ลอง naive baseline ก่อนเสมอ
ถ้า ML ไม่ชนะ baseline = อย่าใช้ (ส่วนใหญ่ก็ไม่ชนะกับ noisy financial data)

---

## ✅ แล้วโปรเจกต์นี้ใช้ได้มั้ย? — ใช้ rule-based แทน

แม้ ML forecasting จะแพ้ baseline แต่ **rule-based system** ทำงานได้ดี:

```python
crisis_signal = (VIX > 30) & (vol_20d > 1.5 * vol_252d_median)
```

ทดสอบกับ COVID-2020:
- ระบบเตือนวันที่ **9 มี.ค. 2020**
- VIX แตะจุดสูงสุด (82.69) วันที่ **16 มี.ค. 2020**
- **เตือนล่วงหน้า 7 วัน** ก่อนตลาดผันผวนสูงสุด

ถ้านักลงทุนเชื่อสัญญาณ → ออกเป็นเงินสด → หลีกเลี่ยงการขาดทุน **~30%** ใน 1 เดือน

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│  Data Layer (data/)                      │
│  ├── market_crawler.py (yfinance)        │
│  └── news_crawler.py (Google News RSS)   │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Engine Layer (engine/)                  │
│  ├── features.py (13 features)           │
│  ├── ml_predictor.py (5 models)          │
│  ├── ai_model.py (FinBERT)               │
│  ├── regime_detector.py                  │
│  ├── backtester.py (Sharpe, MDD, etc.)   │
│  ├── alerts.py (Discord webhook)         │
│  ├── experiment_tracker.py (MLflow)      │
│  └── disk_cache.py (joblib/parquet)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  UI Layer                                │
│  ├── Streamlit app (5 tabs)              │
│  └── Static HTML report (GitHub Pages)   │
└─────────────────────────────────────────┘
```

---

## 🎬 Live Demo

ลองเปิดดูได้ที่: https://oomNoNe.github.io/black-swan-indicator/

- 🌡️ VIX history + 7-day AI forecast
- 🎬 Animated timeline (play button)
- 🌐 3D scatter (หมุนได้)
- 🥇 Model comparison ที่ Naive ชนะ
- 🦠 COVID-2020 case study
- 💰 Backtest with transaction costs

---

## 📚 บทเรียนสรุป

1. **🥇 Always test naive baselines first** — ถ้า ML ไม่ชนะ persistence model = อย่าใช้
2. **🧪 Walk-forward validation** is mandatory สำหรับ time-series (ไม่ใช่ shuffle split)
3. **💰 Transaction costs ต้องคิด** — Sharpe ratio บน paper อาจหายเมื่อ trade จริง
4. **🔍 SHAP > feature_importance** — fair attribution จาก game theory
5. **📊 Rule-based อาจชนะ ML** สำหรับ rare events (เพราะ ML ขาด training data)
6. **🎨 Streamlit สำหรับ interactive, static HTML สำหรับ portfolio sharing**

---

## 🛠️ Tech Used

- Python 3.12 · pandas · numpy
- scikit-learn · XGBoost · LightGBM · PyTorch (LSTM)
- Hugging Face Transformers (FinBERT) · SHAP · MLflow
- Streamlit · Plotly · yfinance
- pytest · GitHub Actions · Docker

---

## 🙏 ขอบคุณที่อ่าน

ถ้าคุณกำลังเริ่มทำโปรเจกต์ ML สาย finance — อย่าลืม **เทียบกับ naive baseline ก่อน**
จะช่วยประหยัดเวลาเทรน + เห็นความจริง earlier

ติดตามได้ที่ [@oomNoNe](https://github.com/oomNoNe) · MIT License

---

*Tags: #MachineLearning #Python #Finance #DataScience #XGBoost #Streamlit #TimeSeriesForecasting #RandomWalk*
