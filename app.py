import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# ---------------------------------------------------------
# Python Path Setup
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.market_crawler import get_current_vix, fetch_historical_data
from data.news_crawler import fetch_financial_news
from engine.ai_model import SentimentAnalyzer
from engine.backtester import run_advanced_backtest, calculate_professional_metrics
from engine.regime_detector import classify_market_regime, dynamic_risk_equation
from engine.ml_predictor import VIXForecaster
from ui.components import (
    draw_gauge_chart, color_sentiment, draw_backtest_chart,
    draw_vix_history_chart, draw_feature_importance,
    draw_equity_curve_chart, draw_drawdown_chart, draw_sentiment_donut,
)

# ==========================================================
# 📚 GLOSSARY — คำอธิบายศัพท์เทคนิคทั้งหมด (ใช้ใน tooltip + sidebar)
# ==========================================================
GLOSSARY = {
    "VIX": "ดัชนีความกลัวของตลาดสหรัฐ (Volatility Index) — วัดความคาดหวังของนักลงทุนต่อความผันผวนใน 30 วันข้างหน้า | < 20 = ตลาดสงบ, 20-30 = กังวล, > 40 = วิกฤต",
    "NLP": "Natural Language Processing — สาขา AI ที่สอนคอมพิวเตอร์ให้เข้าใจภาษาของมนุษย์ (เช่น อ่านข่าวแล้วบอกว่าเป็นเชิงบวก/ลบ)",
    "ML": "Machine Learning — การสอนคอมพิวเตอร์ให้เรียนรู้รูปแบบจากข้อมูลในอดีต เพื่อทำนายอนาคต",
    "Sharpe Ratio": "วัดผลตอบแทนต่อความเสี่ยง — ยิ่งสูงยิ่งดี | > 1 = ดี, > 2 = ดีมาก, < 0 = ขาดทุน",
    "Max Drawdown": "การขาดทุนสะสมสูงสุดจากยอด — บอกว่า 'ถ้าซื้อจุดสูงสุด แล้วขายจุดต่ำสุด จะขาดทุนกี่ %'",
    "Win Rate": "อัตราชนะ — เปอร์เซ็นต์วันที่ได้กำไร เทียบกับวันที่เทรดทั้งหมด",
    "Profit Factor": "อัตราส่วนกำไรรวมต่อขาดทุนรวม | > 1 = กำไร, > 1.5 = ดี, > 2 = ดีมาก",
    "SMA": "Simple Moving Average — ค่าเฉลี่ยราคาย้อนหลัง N วัน ใช้ดูแนวโน้ม",
    "Golden Cross": "เส้น SMA สั้น (50) ตัดขึ้นเหนือเส้น SMA ยาว (200) → สัญญาณตลาดขาขึ้น",
    "Death Cross": "เส้น SMA สั้น (50) ตัดลงใต้เส้น SMA ยาว (200) → สัญญาณตลาดขาลง",
    "Volatility": "ความผันผวน — วัดว่าราคาขึ้น-ลงรุนแรงแค่ไหน | สูง = เสี่ยง, ต่ำ = นิ่ง",
    "Buy & Hold": "กลยุทธ์ซื้อแล้วถือยาว — ไม่ขายไม่ว่าตลาดจะขึ้นหรือลง",
    "Backtest": "การจำลองกลยุทธ์การลงทุนกับข้อมูลในอดีต เพื่อดูว่าถ้าใช้กลยุทธ์นี้จริงจะเป็นยังไง",
    "Regime": "สภาพตลาด — แบ่งเป็น Trending Bull (ขาขึ้น) / Ranging (ออกข้าง) / Panic (วิกฤต)",
    "Sentiment": "อารมณ์ตลาด — วิเคราะห์จากข่าวว่าคนรู้สึก Positive / Neutral / Negative",
    "FinBERT": "โมเดล AI ที่ฝึกมาเข้าใจภาษาทางการเงินโดยเฉพาะ (พัฒนาโดย ProsusAI)",
    "XGBoost": "โมเดล ML ตระกูล Gradient Boosting — เก่งกับข้อมูลแบบตาราง (tabular data) ใช้กันมากใน Kaggle",
    "Lag Features": "การใช้ค่าในอดีต (เช่น VIX เมื่อ 1, 3, 7 วันก่อน) มาทำนายอนาคต",
    "R²": "R-squared — วัดว่าโมเดลอธิบายข้อมูลได้ดีแค่ไหน | 1.0 = สมบูรณ์แบบ, 0 = แย่เท่าเดาค่าเฉลี่ย, ติดลบ = แย่กว่าเดามั่ว",
    "S&P 500": "ดัชนีตลาดหุ้นสหรัฐที่รวม 500 บริษัทใหญ่ที่สุด — เป็น benchmark ตลาดอเมริกา",
}

# ==========================================================
# Page Config
# ==========================================================
st.set_page_config(
    page_title="Black Swan Risk Indicator",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ปรับสไตล์ทั้งหน้าเว็บ — ทำให้ดูพรีเมียม + อ่านง่าย
st.markdown(
    """
    <style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.04);
        padding: 16px 18px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 600; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.85; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        background-color: rgba(255,255,255,0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,71,87,0.15) !important;
        border-bottom: 2px solid #ff4757;
    }

    /* Section headers */
    h2 { border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px; }

    /* Info boxes */
    .term-box {
        background: rgba(31, 119, 180, 0.1);
        border-left: 3px solid #1f77b4;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# HEADER + INTRO BANNER
# ==========================================================
st.title("🚨 Black Swan Risk Indicator")
st.caption("ระบบเตือนภัยล่วงหน้าสำหรับวิกฤตการเงิน · ขับเคลื่อนด้วย AI + Quant Analysis")

# Intro banner สำหรับคนเข้ามาครั้งแรก
with st.expander("👋 มาครั้งแรก? อ่านนี่ก่อน (30 วินาที)", expanded=False):
    intro_col1, intro_col2 = st.columns(2)
    with intro_col1:
        st.markdown(
            """
            ### 🎯 Dashboard นี้ทำอะไร?
            ตรวจจับ**สัญญาณเตือนวิกฤตการเงิน**ก่อนเกิด โดยรวม 3 มุมมอง:

            1. 📰 **ข่าวการเงินทั่วโลก** — วิเคราะห์อารมณ์ด้วย AI
            2. 📊 **ตัวเลขตลาดจริง** — VIX, ความผันผวน, ดัชนีหุ้น
            3. 🤖 **โมเดลพยากรณ์** — ทำนาย VIX ล่วงหน้า 7 วัน

            แล้วรวมเป็น **Crisis Risk Score** (0-100)
            """
        )
    with intro_col2:
        st.markdown(
            """
            ### 🗺️ วิธีอ่าน
            - 🟢 **0-40** = ปลอดภัย (ลงทุนได้ปกติ)
            - 🟠 **40-70** = ระวัง (จับตา)
            - 🔴 **70-100** = วิกฤต (พิจารณาลดความเสี่ยง)

            ### 📑 มี 3 หน้า (Tabs)
            - **Live Dashboard** — สถานะ ณ ปัจจุบัน
            - **AI Prediction** — พยากรณ์ VIX
            - **Backtest** — ทดสอบกลยุทธ์ย้อนหลัง

            > 💡 *ทุก metric ที่มีไอคอน ❓ — เอาเมาส์ไปจ่อจะเห็นคำอธิบาย*
            """
        )

# ==========================================================
# Cached loaders
# ==========================================================
@st.cache_resource
def load_sentiment_ai():
    return SentimentAnalyzer()


@st.cache_data(ttl=3600)
def load_historical_data(years=5):
    return fetch_historical_data(years=years)


@st.cache_resource
def load_ml_predictor(_historical_df):
    forecaster = VIXForecaster()
    if _historical_df is not None and not _historical_df.empty:
        forecaster.train_model(_historical_df)
    return forecaster


# ==========================================================
# SIDEBAR — Global Controls + Glossary
# ==========================================================
with st.sidebar:
    st.header("⚙️ ตั้งค่า")

    years_lookback = st.slider(
        "ข้อมูลย้อนหลัง (ปี)",
        min_value=2, max_value=10, value=5, step=1,
        help="จำนวนปีของข้อมูล S&P 500 + VIX ที่จะโหลดมาวิเคราะห์ (ยิ่งมากยิ่งช้า แต่แม่นกว่า)"
    )

    st.divider()
    st.subheader("🎯 พารามิเตอร์ Backtest")
    vix_threshold = st.slider(
        "VIX Threshold",
        min_value=15.0, max_value=50.0, value=30.0, step=1.0,
        help="ระดับ VIX ที่ถือว่าตลาดเริ่มกลัว (ปกติใช้ 30 — อิงจากวิกฤต 2008 ที่ VIX แตะ 80)"
    )
    vol_multiplier = st.slider(
        "Vol Spike Multiplier",
        min_value=1.0, max_value=3.0, value=1.5, step=0.1,
        help="ค่า volatility 20 วัน ต้องสูงกว่ามัธยฐาน 252 วัน กี่เท่า ถึงจะถือว่าผิดปกติ"
    )

    st.divider()
    st.subheader("📰 ข่าวการเงิน")
    news_lookback = st.slider(
        "จำนวนข่าวที่วิเคราะห์",
        min_value=5, max_value=20, value=10, step=1,
        help="ดึงข่าวล่าสุดจาก Google News กี่ข่าวมาวิเคราะห์ sentiment"
    )

    st.divider()

    # 📚 GLOSSARY — รวมศัพท์ทั้งหมด
    with st.expander("📚 พจนานุกรมศัพท์ (Glossary)", expanded=False):
        st.caption("คลิกชื่อศัพท์เพื่อขยายดูคำอธิบาย")
        for term, definition in GLOSSARY.items():
            with st.popover(f"📖 {term}"):
                st.markdown(f"**{term}**")
                st.write(definition)

    st.divider()
    st.caption(f"🕐 อัพเดท: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("📊 ข้อมูล: yfinance + Google News RSS")
    st.caption("🤖 โมเดล: FinBERT + XGBoost")
    st.caption("🔗 [GitHub Repo](https://github.com/oomNoNe/black-swan-indicator)")

# ==========================================================
# Load Data + Models
# ==========================================================
with st.spinner("⏳ กำลังโหลดข้อมูลตลาด + เทรนโมเดล AI..."):
    sentiment_ai = load_sentiment_ai()
    hist_data = load_historical_data(years=years_lookback)
    ml_forecaster = load_ml_predictor(hist_data)


# ==========================================================
# TOP KPI ROW — Always visible (มี tooltip ไทยทุกอัน)
# ==========================================================
current_vix = get_current_vix()
current_regime = "Unknown"
predicted_vix_top = None

if hist_data is not None and not hist_data.empty:
    current_regime = classify_market_regime(hist_data)
    pred = ml_forecaster.predict_vix(hist_data)
    if not isinstance(pred, str) and not pd.isna(pred):
        predicted_vix_top = pred

st.markdown("### 📍 สถานะปัจจุบัน")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "🌡️ VIX ปัจจุบัน",
        f"{current_vix:.2f}" if current_vix else "N/A",
        help=f"VIX = {GLOSSARY['VIX']}"
    )

with kpi2:
    delta_pred = (predicted_vix_top - current_vix) if (predicted_vix_top and current_vix) else None
    st.metric(
        "🔮 พยากรณ์ AI (7 วัน)",
        f"{predicted_vix_top:.2f}" if predicted_vix_top else "N/A",
        delta=f"{delta_pred:+.2f}" if delta_pred is not None else None,
        delta_color="inverse",
        help="โมเดล XGBoost ทำนายค่า VIX ล่วงหน้า 7 วันทำการ | ลูกศรแดง = VIX จะขึ้น (อันตราย), เขียว = VIX จะลง (สงบ)"
    )

with kpi3:
    regime_emoji = {"Trending Bull": "📈", "Panic": "🔥", "Ranging": "⚖️", "Unknown": "❓"}
    regime_desc = {
        "Trending Bull": "ตลาดขาขึ้น (ปลอดภัย)",
        "Panic": "ตลาดวิกฤต (อันตราย)",
        "Ranging": "ตลาดออกข้าง (เฝ้าระวัง)",
        "Unknown": "ข้อมูลไม่พอ",
    }
    st.metric(
        "🎭 สภาพตลาด",
        f"{regime_emoji.get(current_regime, '❓')} {current_regime}",
        help=f"{regime_desc.get(current_regime)} | คำนวณจาก SMA-50/200 + ความผันผวน 20 วัน"
    )

with kpi4:
    if hist_data is not None and not hist_data.empty:
        recent_close = hist_data['Close'].iloc[-1]
        prev_close = hist_data['Close'].iloc[-2]
        sp_change = ((recent_close - prev_close) / prev_close) * 100
        st.metric(
            "📊 S&P 500",
            f"{recent_close:,.0f}",
            delta=f"{sp_change:+.2f}%",
            help=f"S&P 500 = {GLOSSARY['S&P 500']}"
        )
    else:
        st.metric("S&P 500", "N/A")

st.divider()

# ==========================================================
# TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Live Risk Dashboard",
    "🤖 AI Regime & Prediction",
    "📈 Advanced Quant Backtest",
])

# ==========================================
# TAB 1: Live Risk Dashboard
# ==========================================
with tab1:
    st.subheader("📊 ภาพรวมความเสี่ยงตลาดแบบเรียลไทม์")
    st.caption("รวมข้อมูลตัวเลขตลาด + ข่าวการเงิน → ออกมาเป็น Crisis Risk Score ตัวเดียว")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 📐 ตัวชี้วัดทางเทคนิค")

        if hist_data is not None and not hist_data.empty and len(hist_data) >= 200:
            sma_50 = hist_data['Close'].rolling(50).mean().iloc[-1]
            sma_200 = hist_data['Close'].rolling(200).mean().iloc[-1]
            vol_20_ann = hist_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

            st.metric(
                "📈 SMA-50",
                f"{sma_50:,.0f}",
                help=f"{GLOSSARY['SMA']} | SMA-50 = ค่าเฉลี่ยราคาปิด 50 วันล่าสุด ใช้ดูแนวโน้มระยะกลาง"
            )
            st.metric(
                "📉 SMA-200",
                f"{sma_200:,.0f}",
                help="ค่าเฉลี่ยราคาปิด 200 วัน ใช้ดูแนวโน้มระยะยาว — เป็นเส้นที่นักลงทุนทั่วโลกเฝ้าดู"
            )
            st.metric(
                "⚡ ความผันผวน 20 วัน (annualized)",
                f"{vol_20_ann:.2f}%",
                help=f"{GLOSSARY['Volatility']} | คูณ √252 เพื่อแปลงเป็นรายปี | < 15% = ปกติ, > 30% = ผันผวนสูง"
            )

            golden_cross = sma_50 > sma_200
            if golden_cross:
                st.success(f"🟢 **Golden Cross** — {GLOSSARY['Golden Cross']}")
            else:
                st.error(f"🔴 **Death Cross** — {GLOSSARY['Death Cross']}")

        st.markdown("##### 🎭 หลักการแบ่ง Regime")
        st.caption(
            "📈 **Trending Bull**: SMA-50 > SMA-200 + ความผันผวนต่ำ\n\n"
            "🔥 **Panic**: SMA-50 < SMA-200 + ความผันผวนสูง\n\n"
            "⚖️ **Ranging**: นอกเหนือจาก 2 กรณีข้างต้น"
        )

    with col_right:
        st.markdown("#### 🌍 Sentiment ข่าวการเงินทั่วโลก")

        with st.spinner("📡 กำลังดึงข่าวล่าสุด..."):
            news_df = fetch_financial_news()

        if news_df is not None and not news_df.empty:
            news_df = news_df.head(news_lookback).copy()
            news_df[['Sentiment', 'Confidence']] = news_df['Headline'].apply(
                lambda x: pd.Series(sentiment_ai.analyze(x))
            )
            sentiment_scores = {"NEGATIVE": 100, "NEUTRAL": 50, "POSITIVE": 0}
            news_df['Risk_Score'] = news_df['Sentiment'].map(sentiment_scores).fillna(50)
            avg_news_risk = news_df['Risk_Score'].mean()

            market_risk = min((current_vix / 40.0) * 100, 100.0) if current_vix else 50.0
            final_risk = dynamic_risk_equation(avg_news_risk, market_risk, current_regime)

            gauge_col, donut_col = st.columns([2, 1])
            with gauge_col:
                st.plotly_chart(draw_gauge_chart(final_risk), use_container_width=True)
            with donut_col:
                st.plotly_chart(draw_sentiment_donut(news_df), use_container_width=True)

            # Risk breakdown
            breakdown_col1, breakdown_col2 = st.columns(2)
            with breakdown_col1:
                st.metric(
                    "📰 News Risk เฉลี่ย", f"{avg_news_risk:.1f}/100",
                    help="คะแนนความเสี่ยงเฉลี่ยจากข่าวทั้งหมด — Negative=100, Neutral=50, Positive=0"
                )
            with breakdown_col2:
                st.metric(
                    "📊 Market Risk (จาก VIX)", f"{market_risk:.1f}/100",
                    help="แปลง VIX เป็นสเกล 0-100 (VIX=40 → 100 คะแนน) | สะท้อนความกลัวเชิงปริมาณ"
                )

            with st.expander("💡 Risk Score คำนวณยังไง?", expanded=False):
                st.markdown(
                    f"""
                    **สูตร**: `Risk Score = w_news × News_Risk + w_market × Market_Risk`

                    น้ำหนัก (w) ปรับตาม Regime อัตโนมัติ:
                    - 📈 **Trending Bull** → w_news=0.3, w_market=0.7 *(ตลาดดี เชื่อตัวเลขมากกว่าข่าว)*
                    - 🔥 **Panic** → w_news=0.7, w_market=0.3 *(วิกฤต ข่าวสำคัญที่สุด)*
                    - ⚖️ **Ranging** → w_news=0.5, w_market=0.5

                    **ตอนนี้**: Regime = `{current_regime}`,
                    News Risk = `{avg_news_risk:.1f}`,
                    Market Risk = `{market_risk:.1f}`
                    → **Final Score = {final_risk:.1f}**
                    """
                )

            st.markdown("##### 📑 หัวข้อข่าวล่าสุด")
            display_df = news_df[['Publish_Date', 'Headline', 'Sentiment', 'Confidence']].copy()
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.0f}")
            st.dataframe(
                display_df.style.map(color_sentiment, subset=['Sentiment']),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("⚠️ ดึงข่าวไม่สำเร็จ — กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")

# ==========================================
# TAB 2: AI Regime & Prediction
# ==========================================
with tab2:
    st.subheader("🤖 พยากรณ์ VIX ล่วงหน้า 7 วันด้วย Machine Learning")
    st.caption("ใช้โมเดล XGBoost ที่ฝึกบน lag features ของ VIX และ S&P 500 returns")

    if hist_data is not None and not hist_data.empty and predicted_vix_top is not None:
        # Top metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "VIX ปัจจุบัน", f"{current_vix:.2f}" if current_vix else "N/A",
                help=GLOSSARY['VIX']
            )
        with m2:
            st.metric(
                "พยากรณ์ (7 วัน)", f"{predicted_vix_top:.2f}",
                delta=f"{predicted_vix_top - current_vix:+.2f}" if current_vix else None,
                delta_color="inverse",
                help="ค่า VIX ที่โมเดลคาดว่าจะเป็นในอีก 7 วันทำการ"
            )
        with m3:
            train_r2 = ml_forecaster.train_score if ml_forecaster.train_score is not None else 0
            st.metric(
                "Train R²", f"{train_r2:.3f}",
                help=f"{GLOSSARY['R²']} | นี่คือคะแนนบนข้อมูลที่โมเดลเคยเห็น (in-sample)"
            )
        with m4:
            test_r2 = ml_forecaster.test_score if ml_forecaster.test_score is not None else 0
            st.metric(
                "Test R² (out-of-sample)", f"{test_r2:.3f}",
                help="คะแนนบนข้อมูลที่โมเดลไม่เคยเห็น = ของจริง! | VIX พยากรณ์ยากมาก R² ใกล้ 0 ถือว่าปกติ"
            )

        st.divider()

        # VIX history + forecast chart
        st.plotly_chart(
            draw_vix_history_chart(hist_data, predicted_vix=predicted_vix_top, lookback_days=180),
            use_container_width=True
        )

        st.caption(
            "💡 **วิธีอ่านกราฟ**: เส้นน้ำเงิน = VIX จริงในอดีต | เส้นส้มประ = ค่าเฉลี่ย 20 วัน (ช่วยมองเทรนด์) | "
            "ดาวแดง = จุดที่โมเดลทำนายว่า VIX จะเป็นในอีก 7 วัน"
        )

        # Feature importance + interpretation
        fi_col, interp_col = st.columns([1, 1])

        with fi_col:
            fi_fig = draw_feature_importance(ml_forecaster)
            if fi_fig is not None:
                st.plotly_chart(fi_fig, use_container_width=True)
            st.caption(
                "📊 **Feature Importance** = โมเดลให้น้ำหนักแต่ละ feature เท่าไหร่ "
                "(แท่งยาว = สำคัญมาก)"
            )

        with interp_col:
            st.markdown("##### 🧠 การตีความผลพยากรณ์")

            direction = (
                "📈 **VIX จะขึ้น** — ตลาดอาจผันผวนมากขึ้น เตรียมระวัง"
                if predicted_vix_top > current_vix
                else "📉 **VIX จะลง** — สถานการณ์น่าจะคลี่คลาย"
            )
            st.markdown(direction)

            crisis_level = (
                "🔴 **HIGH** — โอกาสเกิดวิกฤตสูง" if predicted_vix_top > 30
                else "🟠 **MEDIUM** — เฝ้าระวัง" if predicted_vix_top > 20
                else "🟢 **LOW** — ปลอดภัย"
            )
            st.markdown(f"**ระดับความเสี่ยงวิกฤต**: {crisis_level}")

            st.markdown("---")
            st.caption(
                "⚠️ **ข้อควรระวัง**: VIX เป็นตัวแปรที่พยากรณ์ยากที่สุดตัวหนึ่งในโลกการเงิน "
                "แม้โมเดลที่ดีก็มักได้ R² ใกล้ 0 ใช้เป็น *สัญญาณทิศทาง* เท่านั้น อย่ายึดเป็นความจริงสัมบูรณ์"
            )

        st.divider()

        # ============================================
        # 🎓 ทำไมเลือกโมเดลเหล่านี้? — Section ใหม่
        # ============================================
        st.markdown("### 🎓 ทำไมต้องเลือกโมเดลเหล่านี้?")
        st.caption("เหตุผลในการเลือกแต่ละโมเดล พร้อมข้อดี-ข้อเสีย")

        why_col1, why_col2, why_col3 = st.columns(3)

        with why_col1:
            with st.container(border=True):
                st.markdown("#### 🤖 FinBERT")
                st.caption("สำหรับวิเคราะห์ Sentiment ข่าว")
                st.markdown("**✅ ข้อดี**")
                st.markdown(
                    "- ฝึกบน corpus การเงินโดยเฉพาะ → เข้าใจศัพท์เช่น 'bearish', 'guidance', 'hawkish'\n"
                    "- ความแม่นยำสูงกว่า BERT ทั่วไป ~15% สำหรับข่าวการเงิน\n"
                    "- เปิด open-source ฟรี (ProsusAI)\n"
                    "- เร็วพอใช้บน CPU"
                )
                st.markdown("**❌ ข้อเสีย**")
                st.markdown(
                    "- รองรับแค่ภาษาอังกฤษ\n"
                    "- ฝึกปี 2019 → ไม่รู้ศัพท์ใหม่ (เช่น GameStop saga)\n"
                    "- 110M parameters → ใช้ RAM ~440MB\n"
                    "- อาจ bias จาก training data"
                )

        with why_col2:
            with st.container(border=True):
                st.markdown("#### 🌲 XGBoost")
                st.caption("สำหรับพยากรณ์ VIX")
                st.markdown("**✅ ข้อดี**")
                st.markdown(
                    "- ชนะ Kaggle competitions จำนวนมาก → พิสูจน์แล้วว่าเก่งกับ tabular data\n"
                    "- เร็วมาก (เทรนใน < 1 วินาที)\n"
                    "- ตีความได้ผ่าน feature importance\n"
                    "- รับมือกับ missing values ได้\n"
                    "- ไม่ต้อง scale features"
                )
                st.markdown("**❌ ข้อเสีย**")
                st.markdown(
                    "- ไม่มี memory ของเวลา → ต้องสร้าง lag features เอง\n"
                    "- Overfit ง่ายถ้าไม่ tune\n"
                    "- ไม่เก่งกับ extrapolation (ทำนายค่าที่ไม่เคยเห็น)\n"
                    "- เทียบกับ Deep Learning แล้วอาจแพ้บนข้อมูลใหญ่มากๆ"
                )

        with why_col3:
            with st.container(border=True):
                st.markdown("#### 📊 SMA + Vol (Regime)")
                st.caption("สำหรับจำแนกสภาพตลาด")
                st.markdown("**✅ ข้อดี**")
                st.markdown(
                    "- เรียบง่าย เข้าใจง่าย ตีความได้ทันที\n"
                    "- ไม่ต้อง hyperparameter tuning\n"
                    "- เป็นมาตรฐานที่ทุกกองทุนใช้\n"
                    "- คำนวณเร็ว (ไม่ใช้ ML)"
                )
                st.markdown("**❌ ข้อเสีย**")
                st.markdown(
                    "- **Lagging indicator** — SMA-200 ตอบสนองช้า\n"
                    "- ใช้ threshold แบบ binary (อาจพลาดช่วง transition)\n"
                    "- ไม่ใช้ข้อมูลข่าว/macro\n"
                    "- ทางเลือกที่ดีกว่า: Hidden Markov Model หรือ Bayesian regime switching"
                )

        with st.expander("🔬 อยากรู้ลึกกว่านี้? อ่านที่มาของแต่ละโมเดล", expanded=False):
            st.markdown(
                """
                **FinBERT** — *Araci, 2019. "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"*
                - Hugging Face: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
                - ฝึกบน Reuters TRC2 corpus + Financial PhraseBank

                **XGBoost** — *Chen & Guestrin, 2016. "XGBoost: A Scalable Tree Boosting System"*
                - หลักการ: รวม decision trees อ่อนๆ หลายตัวให้กลายเป็นโมเดลที่แข็งแกร่ง (ensemble)
                - ใช้ gradient descent บน loss function ของ tree

                **Regime Detection** — แนวคิดมาจาก *Hamilton (1989)* regime-switching models
                - งานนี้ใช้แบบ rule-based แทน statistical (ตามหลัก Occam's Razor)
                - การเปลี่ยนแปลงใน roadmap: ไปใช้ Hidden Markov Model (HMM) ในอนาคต
                """
            )
    else:
        st.error("ข้อมูลไม่พอสำหรับ ML — ตรวจสอบการเชื่อมต่อ yfinance")

# ==========================================
# TAB 3: Advanced Quant Backtest
# ==========================================
with tab3:
    st.subheader("📈 ทดสอบกลยุทธ์ย้อนหลัง (Historical Backtest)")
    st.caption(
        f"**สูตรสัญญาณ**: VIX > {vix_threshold:.0f} **AND** ความผันผวน 20 วัน > {vol_multiplier:.1f}× มัธยฐาน 252 วัน"
    )

    with st.expander("📖 Backtest คืออะไร? ทำไมต้องทำ?", expanded=False):
        st.markdown(
            f"""
            **{GLOSSARY['Backtest']}**

            **ตัวอย่าง**: ถ้ามีเงิน 1 ล้านบาท แล้วใช้กลยุทธ์นี้ตั้งแต่ {years_lookback} ปีที่แล้ว
            จะได้เงินเท่าไหร่? เทียบกับซื้อแล้วถือเฉยๆ (Buy & Hold) อันไหนดีกว่า?

            **กลยุทธ์นี้ทำงานยังไง**:
            1. ทุกวันคำนวณว่า "VIX สูง + ความผันผวนพุ่ง" หรือไม่
            2. ถ้าใช่ → ดึงเงินสดออกจากตลาด (avoid crash)
            3. ถ้าไม่ใช่ → ถือหุ้นตามปกติ
            """
        )

    if hist_data is not None and not hist_data.empty:
        bt_data = hist_data.copy()
        bt_data['Vol_20'] = bt_data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        bt_data['Vol_Median_252'] = bt_data['Vol_20'].rolling(252, min_periods=60).median()
        bt_data['Risk_Signal'] = (
            (bt_data['VIX'] > vix_threshold) &
            (bt_data['Vol_20'] > bt_data['Vol_Median_252'] * vol_multiplier)
        ).astype(int)

        n_signals = int(bt_data['Risk_Signal'].sum())
        pct_in_cash = (n_signals / len(bt_data)) * 100

        metrics = run_advanced_backtest(bt_data)

        if metrics:
            st.markdown("#### 📊 ผลการ Backtest")

            strat = metrics["Black_Swan_Strategy"]
            base = metrics["Baseline_Buy_Hold"]

            # KPI row comparing strategy vs baseline
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                diff = strat["Sharpe Ratio"] - base["Sharpe Ratio"]
                st.metric(
                    "Sharpe Ratio", f"{strat['Sharpe Ratio']:.2f}",
                    delta=f"{diff:+.2f} vs B&H",
                    help=GLOSSARY['Sharpe Ratio']
                )
            with m2:
                diff = strat["Max Drawdown (%)"] - base["Max Drawdown (%)"]
                st.metric(
                    "Max Drawdown", f"{strat['Max Drawdown (%)']:.2f}%",
                    delta=f"{diff:+.2f}% vs B&H", delta_color="inverse",
                    help=GLOSSARY['Max Drawdown']
                )
            with m3:
                st.metric(
                    "Win Rate", f"{strat['Win Rate (%)']:.2f}%",
                    help=GLOSSARY['Win Rate']
                )
            with m4:
                pf = strat['Profit Factor']
                pf_str = f"{pf:.2f}" if not np.isnan(pf) else "∞"
                st.metric(
                    "Profit Factor", pf_str,
                    help=GLOSSARY['Profit Factor']
                )

            st.divider()

            # Side-by-side comparison
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                st.markdown("##### 🛡️ Black Swan Strategy")
                st.json(strat)
                st.caption(f"🔔 ส่งสัญญาณ {n_signals} ครั้ง ({pct_in_cash:.1f}% ของวันทั้งหมดอยู่ในเงินสด)")
            with cmp_col2:
                st.markdown(f"##### 📊 Baseline ({GLOSSARY['Buy & Hold'].split('—')[0].strip()})")
                st.json(base)
                st.caption("📈 ถือหุ้นตลอด — ไม่มีช่วงพักในเงินสด")

            st.divider()

            # Charts
            st.plotly_chart(draw_equity_curve_chart(bt_data), use_container_width=True)
            st.caption(
                "💡 **วิธีอ่าน**: เส้นเขียว = กลยุทธ์เรา | เส้นเทา = Buy & Hold | "
                "ค่า 1.5 = เงินโต 50% | ถ้าเส้นเขียวอยู่บนเส้นเทา = กลยุทธ์เราชนะ"
            )

            dd_col, sig_col = st.columns([1, 1])
            with dd_col:
                st.plotly_chart(draw_drawdown_chart(bt_data), use_container_width=True)
                st.caption("📉 พื้นที่แดง = ช่วงขาดทุนสะสม | ยิ่งตื้นยิ่งดี")
            with sig_col:
                st.plotly_chart(draw_backtest_chart(bt_data), use_container_width=True)
                st.caption("🚨 จุดแดง = เวลาที่ระบบส่งสัญญาณเตือนวิกฤต")

            # Honest insight box
            with st.expander("💡 อ่านผลยังไงให้ถูกต้อง", expanded=False):
                st.markdown(
                    f"""
                    ### ทำความเข้าใจผลลัพธ์

                    - **Sharpe Ratio**: {GLOSSARY['Sharpe Ratio']}
                      → กลยุทธ์ crash-avoidance มัก Sharpe ต่ำกว่า B&H นิดหน่อย เพราะยอม sacrifice กำไรช่วงตลาดดี

                    - **Max Drawdown**: {GLOSSARY['Max Drawdown']}
                      → กลยุทธ์ที่ดีควรมี drawdown น้อยกว่า B&H อย่างชัดเจน (ถ้าไม่ลด → กลยุทธ์ใช้ไม่ได้)

                    - **Profit Factor**: {GLOSSARY['Profit Factor']}

                    ### ⚠️ ข้อจำกัดสำคัญ
                    - **ไม่คิด transaction cost** — ในชีวิตจริงต้องบวก ~0.05-0.10% ต่อการเทรด
                    - **ไม่คิด slippage** — เวลาขายในช่วงตลาด crash จริง ราคาที่ได้อาจแย่กว่าที่คำนวณ
                    - **Survivorship bias** — S&P 500 มีการ rebalance ตลอด เราใช้ index ปัจจุบัน
                    - **Past ≠ Future** — ผลในอดีตไม่การันตีอนาคต ตลาดอาจเปลี่ยน regime
                    """
                )
        else:
            st.warning("คำนวณ backtest metrics ไม่ได้ — ข้อมูลอาจไม่เพียงพอ")
    else:
        st.error("ข้อมูลย้อนหลังไม่พร้อม — ตรวจสอบการเชื่อมต่อ yfinance")
