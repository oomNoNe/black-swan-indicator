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

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Black Swan Risk Indicator",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stMetric { background-color: rgba(255,255,255,0.04); padding: 12px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚨 Black Swan Risk Indicator (Quant Edition)")
st.markdown(
    "*An AI-powered & Quant-driven Early Warning System for Financial Crises — "
    "combining FinBERT sentiment, XGBoost forecasting, regime detection, and backtested signals.*"
)


# ---------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# SIDEBAR — Global Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    years_lookback = st.slider(
        "Historical Data (Years)",
        min_value=2, max_value=10, value=5, step=1,
        help="Number of years of S&P 500 + VIX data to load"
    )

    st.divider()
    st.subheader("Backtest Signal")
    vix_threshold = st.slider(
        "VIX Threshold",
        min_value=15.0, max_value=50.0, value=30.0, step=1.0,
        help="VIX level above which the market is considered fearful"
    )
    vol_multiplier = st.slider(
        "Vol Spike Multiplier",
        min_value=1.0, max_value=3.0, value=1.5, step=0.1,
        help="20-day vol must exceed this × the 252-day median"
    )

    st.divider()
    st.subheader("News Sentiment")
    news_lookback = st.slider(
        "Headlines to Analyze",
        min_value=5, max_value=20, value=10, step=1,
    )

    st.divider()
    st.caption(f"🕐 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.caption("📊 Data: yfinance + Google News RSS")
    st.caption("🤖 Models: FinBERT + XGBoost")

# ---------------------------------------------------------
# Load Data + Models
# ---------------------------------------------------------
with st.spinner("Loading market data & training models..."):
    sentiment_ai = load_sentiment_ai()
    hist_data = load_historical_data(years=years_lookback)
    ml_forecaster = load_ml_predictor(hist_data)


# ---------------------------------------------------------
# TOP KPI ROW — Always visible
# ---------------------------------------------------------
current_vix = get_current_vix()
current_regime = "Unknown"
predicted_vix_top = None

if hist_data is not None and not hist_data.empty:
    current_regime = classify_market_regime(hist_data)
    pred = ml_forecaster.predict_vix(hist_data)
    if not isinstance(pred, str) and not pd.isna(pred):
        predicted_vix_top = pred

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Current VIX",
        f"{current_vix:.2f}" if current_vix else "N/A",
        help="Real-time VIX (Fear Gauge). Normal: <20, Crisis: >40"
    )

with kpi2:
    delta_pred = (predicted_vix_top - current_vix) if (predicted_vix_top and current_vix) else None
    st.metric(
        "AI Forecast (7d)",
        f"{predicted_vix_top:.2f}" if predicted_vix_top else "N/A",
        delta=f"{delta_pred:+.2f}" if delta_pred is not None else None,
        delta_color="inverse",
        help="XGBoost prediction of VIX 7 trading days ahead"
    )

with kpi3:
    regime_emoji = {"Trending Bull": "📈", "Panic": "🔥", "Ranging": "⚖️", "Unknown": "❓"}
    st.metric(
        "Market Regime",
        f"{regime_emoji.get(current_regime, '❓')} {current_regime}",
        help="Detected from SMA-50/200 crossover + 20-day rolling volatility"
    )

with kpi4:
    if hist_data is not None and not hist_data.empty:
        recent_close = hist_data['Close'].iloc[-1]
        prev_close = hist_data['Close'].iloc[-2]
        sp_change = ((recent_close - prev_close) / prev_close) * 100
        st.metric(
            "S&P 500",
            f"{recent_close:,.0f}",
            delta=f"{sp_change:+.2f}%",
            help="Latest S&P 500 close vs previous day"
        )
    else:
        st.metric("S&P 500", "N/A")

st.divider()

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Live Risk Dashboard",
    "🤖 AI Regime & Prediction",
    "📈 Advanced Quant Backtest",
])

# ==========================================
# TAB 1: Live Risk Dashboard
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Regime & Market Indicators")

        if hist_data is not None and not hist_data.empty and len(hist_data) >= 200:
            sma_50 = hist_data['Close'].rolling(50).mean().iloc[-1]
            sma_200 = hist_data['Close'].rolling(200).mean().iloc[-1]
            vol_20_ann = hist_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

            st.metric("SMA-50", f"{sma_50:,.0f}")
            st.metric("SMA-200", f"{sma_200:,.0f}")
            st.metric("Realized Vol (20d, annualized)", f"{vol_20_ann:.2f}%")

            golden_cross = sma_50 > sma_200
            cross_msg = "🟢 Golden Cross (bullish)" if golden_cross else "🔴 Death Cross (bearish)"
            st.info(cross_msg)

        st.markdown("##### Regime Logic")
        st.caption(
            "• **Trending Bull**: SMA-50 > SMA-200 & low vol\n\n"
            "• **Panic**: SMA-50 < SMA-200 & high vol\n\n"
            "• **Ranging**: everything else"
        )

    with col_right:
        st.subheader("Global Financial News Sentiment")

        with st.spinner("Fetching latest news..."):
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
                st.metric("News Risk Avg", f"{avg_news_risk:.1f}/100")
            with breakdown_col2:
                st.metric("Market Risk (VIX-based)", f"{market_risk:.1f}/100")

            st.markdown("##### Latest Headlines")
            display_df = news_df[['Publish_Date', 'Headline', 'Sentiment', 'Confidence']].copy()
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.0f}")
            st.dataframe(
                display_df.style.map(color_sentiment, subset=['Sentiment']),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Failed to fetch recent news.")

# ==========================================
# TAB 2: AI Regime & Prediction
# ==========================================
with tab2:
    st.subheader("Predictive ML: 7-Day VIX Forecast")
    st.caption("XGBoost Regressor trained on lagged macro features (VIX lag 1/3/7, S&P returns 1d/5d).")

    if hist_data is not None and not hist_data.empty and predicted_vix_top is not None:
        # Top metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Current VIX", f"{current_vix:.2f}" if current_vix else "N/A")
        with m2:
            st.metric("Predicted (7d)", f"{predicted_vix_top:.2f}",
                      delta=f"{predicted_vix_top - current_vix:+.2f}" if current_vix else None,
                      delta_color="inverse")
        with m3:
            train_r2 = ml_forecaster.train_score if ml_forecaster.train_score is not None else 0
            st.metric("Train R²", f"{train_r2:.3f}")
        with m4:
            test_r2 = ml_forecaster.test_score if ml_forecaster.test_score is not None else 0
            st.metric("Test R² (out-of-sample)", f"{test_r2:.3f}",
                      help="R² < 0 means worse than predicting the mean — VIX is notoriously hard to predict")

        st.divider()

        # VIX history + forecast chart
        st.plotly_chart(
            draw_vix_history_chart(hist_data, predicted_vix=predicted_vix_top, lookback_days=180),
            use_container_width=True
        )

        # Feature importance + interpretation
        fi_col, interp_col = st.columns([1, 1])

        with fi_col:
            fi_fig = draw_feature_importance(ml_forecaster)
            if fi_fig is not None:
                st.plotly_chart(fi_fig, use_container_width=True)

        with interp_col:
            st.markdown("##### 🧠 Model Interpretation")
            st.markdown(
                "**Forecast direction:** "
                + ("📈 **Volatility rising** — caution advised"
                   if predicted_vix_top > current_vix
                   else "📉 **Volatility easing** — calmer conditions ahead")
            )
            st.markdown(
                f"**Crisis probability proxy:** "
                + ("🔴 HIGH" if predicted_vix_top > 30
                   else "🟠 MEDIUM" if predicted_vix_top > 20
                   else "🟢 LOW")
            )
            st.markdown("---")
            st.caption(
                "⚠️ **Disclaimer**: VIX is extremely difficult to forecast — even strong models often have R² near zero. "
                "Treat predictions as a *directional* signal, not absolute truth."
            )
    else:
        st.error("Insufficient historical data to run ML predictions. Check yfinance connection.")

# ==========================================
# TAB 3: Advanced Quant Backtest
# ==========================================
with tab3:
    st.subheader("Historical Backtesting & Strategy Evaluation")
    st.caption(
        f"**Signal Logic**: VIX > {vix_threshold:.0f} **AND** 20-day vol > {vol_multiplier:.1f}× 252-day median vol"
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
            st.markdown("#### 📊 Strategy Metrics")

            strat = metrics["Black_Swan_Strategy"]
            base = metrics["Baseline_Buy_Hold"]

            # KPI row comparing strategy vs baseline
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                diff = strat["Sharpe Ratio"] - base["Sharpe Ratio"]
                st.metric("Sharpe Ratio", f"{strat['Sharpe Ratio']:.2f}",
                          delta=f"{diff:+.2f} vs B&H", delta_color="normal")
            with m2:
                diff = strat["Max Drawdown (%)"] - base["Max Drawdown (%)"]
                st.metric("Max Drawdown", f"{strat['Max Drawdown (%)']:.2f}%",
                          delta=f"{diff:+.2f}% vs B&H", delta_color="inverse")
            with m3:
                st.metric("Win Rate", f"{strat['Win Rate (%)']:.2f}%")
            with m4:
                pf = strat['Profit Factor']
                pf_str = f"{pf:.2f}" if not np.isnan(pf) else "∞"
                st.metric("Profit Factor", pf_str)

            st.divider()

            # Side-by-side comparison table
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                st.markdown("##### 🛡️ Black Swan Strategy")
                st.json(strat)
                st.caption(f"🔔 {n_signals} signals fired ({pct_in_cash:.1f}% of days in cash)")
            with cmp_col2:
                st.markdown("##### 📊 Baseline (Buy & Hold)")
                st.json(base)
                st.caption("📈 Always in market — no cash position")

            st.divider()

            # Charts
            st.plotly_chart(draw_equity_curve_chart(bt_data), use_container_width=True)

            dd_col, sig_col = st.columns([1, 1])
            with dd_col:
                st.plotly_chart(draw_drawdown_chart(bt_data), use_container_width=True)
            with sig_col:
                st.plotly_chart(draw_backtest_chart(bt_data), use_container_width=True)

            # Honest insight box
            with st.expander("💡 How to read these results", expanded=False):
                st.markdown(
                    """
                    - **Sharpe**: higher = better risk-adjusted return. A Sharpe slightly below B&H is *normal*
                      for crash-avoidance strategies because they sacrifice upside in calm periods.
                    - **Max Drawdown**: the strategy *should* show smaller drawdowns if it works.
                    - **Profit Factor**: ratio of gross profit to gross loss. > 1 = profitable.
                    - **⚠️ This backtest assumes zero transaction costs** — a strict realistic test would
                      add ~0.05–0.10% per turnover, which would worsen the strategy's Sharpe.
                    """
                )
        else:
            st.warning("Could not calculate backtest metrics. Data might be insufficient.")
    else:
        st.error("Historical data unavailable for backtesting. Check yfinance connection.")
