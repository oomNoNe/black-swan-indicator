import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# ---------------------------------------------------------
# ป้องกันปัญหา Python Path Hell
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 📥 Imports
from data.market_crawler import get_current_vix, fetch_historical_data
from data.news_crawler import fetch_financial_news
from engine.ai_model import SentimentAnalyzer
from engine.backtester import run_advanced_backtest
from engine.regime_detector import classify_market_regime, dynamic_risk_equation
from engine.ml_predictor import VIXForecaster
from ui.components import draw_gauge_chart, color_sentiment, draw_backtest_chart

# ---------------------------------------------------------
# การตั้งค่าหน้าเพจ Streamlit
# ---------------------------------------------------------
st.set_page_config(page_title="Black Swan Risk Indicator", page_icon="🚨", layout="wide")
st.title("🚨 Black Swan Risk Indicator (Quant Edition)")
st.markdown("An AI-powered & Quant-driven Early Warning System for Financial Crises.")


# ---------------------------------------------------------
# 🧠 โหลด AI Models ลงใน Cache
# ---------------------------------------------------------
@st.cache_resource
def load_sentiment_ai():
    return SentimentAnalyzer()


@st.cache_data(ttl=3600)
def load_historical_data():
    return fetch_historical_data(years=5)


@st.cache_resource
def load_ml_predictor(_historical_df):
    forecaster = VIXForecaster()
    if _historical_df is not None and not _historical_df.empty:
        forecaster.train_model(_historical_df)
    return forecaster


sentiment_ai = load_sentiment_ai()
hist_data = load_historical_data()
ml_forecaster = load_ml_predictor(hist_data)

# ---------------------------------------------------------
# 🗂️ สร้างระบบ Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Live Risk Dashboard", "🤖 AI Regime & Prediction", "📈 Advanced Quant Backtest"])

# ==========================================
# TAB 1: Live Risk Dashboard
# ==========================================
with tab1:
    st.header("Real-Time Market Assessment")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Market Regimes & Indicators")
        current_vix = get_current_vix()

        current_regime = "Unknown"
        if hist_data is not None and not hist_data.empty:
            current_regime = classify_market_regime(hist_data)

        st.metric(label="Current VIX (Fear Gauge)", value=f"{current_vix}" if current_vix is not None else "N/A")
        st.info(f"**Detected Market Regime:** {current_regime}")

    with col2:
        st.subheader("Global Financial News Sentiment")
        news_df = fetch_financial_news()

        if news_df is not None and not news_df.empty:
            news_df[['Sentiment', 'Confidence']] = news_df['Headline'].apply(
                lambda x: pd.Series(sentiment_ai.analyze(x))
            )

            sentiment_scores = {"NEGATIVE": 100, "NEUTRAL": 50, "POSITIVE": 0}
            news_df['Risk_Score'] = news_df['Sentiment'].map(sentiment_scores).fillna(50)
            avg_news_risk = news_df['Risk_Score'].mean()

            market_risk = min((current_vix / 40.0) * 100, 100.0) if current_vix else 50.0
            final_risk = dynamic_risk_equation(avg_news_risk, market_risk, current_regime)

            st.plotly_chart(draw_gauge_chart(final_risk), use_container_width=True)
            st.dataframe(news_df.style.map(color_sentiment, subset=['Sentiment']))
        else:
            st.warning("Failed to fetch recent news.")

# ==========================================
# TAB 2: AI Regime & Prediction
# ==========================================
with tab2:
    st.header("Predictive ML: 7-Day VIX Forecast")
    st.markdown("Uses an XGBoost Regressor trained on lagged macroeconomic features to forecast near-term volatility.")

    if hist_data is not None and not hist_data.empty:
        predicted_vix = ml_forecaster.predict_vix(hist_data)

        # 🛠️ ป้องกัน Error กรณีที่โมเดลไม่ได้ Train หรือคืนค่าเป็น Text
        if isinstance(predicted_vix, str) or pd.isna(predicted_vix):
            st.warning(
                "⚠️ AI Model is uninitialized or lack sufficient data to predict. Please check your internet connection and reload.")
        else:
            # คำนวณส่วนต่าง (Delta) อย่างปลอดภัย
            delta_val = round(predicted_vix - current_vix, 2) if current_vix else None

            st.metric(
                label="Predicted VIX (Next 7 Days)",
                value=f"{predicted_vix}",
                delta=f"{delta_val} from current" if delta_val is not None else None,
                delta_color="inverse"
            )
            st.caption(
                "*Mathematical Intuition: Model uses rolling features (Lag 1, 3, 7 days of VIX and SP500 Returns).*")
    else:
        st.error("Insufficient historical data to run ML predictions. Please check yfinance connection.")

# ==========================================
# TAB 3: Advanced Quant Backtest
# ==========================================
with tab3:
    st.header("Historical Backtesting & Strategy Evaluation")
    st.markdown("Comparing the Black Swan Crisis evasion strategy against a standard Buy & Hold approach.")

    if hist_data is not None and not hist_data.empty:
        # สร้างสำเนาข้อมูลเพื่อไม่ให้กระทบ Tab อื่น
        bt_data = hist_data.copy()

        # 🌟 Real signal: ใช้ VIX threshold + Volatility spike (ไม่ใช่สุ่ม)
        #   - VIX > 30 = ตลาดกลัวสูง (อิงจากค่าประวัติศาสตร์ช่วงวิกฤต 2008/2020)
        #   - Vol spike = ผันผวน 20 วันสูงกว่า median ของช่วง 252 วัน × 1.5
        vix_threshold = 30.0
        bt_data['Vol_20'] = bt_data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        bt_data['Vol_Median_252'] = bt_data['Vol_20'].rolling(252, min_periods=60).median()
        bt_data['Risk_Signal'] = (
            (bt_data['VIX'] > vix_threshold) &
            (bt_data['Vol_20'] > bt_data['Vol_Median_252'] * 1.5)
        ).astype(int)

        st.caption(
            f"*Signal Logic: VIX > {vix_threshold} **AND** rolling 20-day vol > 1.5× rolling 252-day median.*"
        )

        metrics = run_advanced_backtest(bt_data)

        if metrics:
            st.subheader("Strategy Metrics")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.write("**Black Swan System**")
                st.json(metrics["Black_Swan_Strategy"])
            with m_col2:
                st.write("**Baseline (Buy & Hold)**")
                st.json(metrics["Baseline_Buy_Hold"])

            st.plotly_chart(draw_backtest_chart(bt_data), use_container_width=True)
        else:
            st.warning("Could not calculate backtest metrics. Data might be insufficient.")
    else:
        st.error("Historical data unavailable for backtesting. Please check yfinance connection.")