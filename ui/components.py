import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# ==========================================================
# GAUGE CHART — Crisis Risk Score
# ==========================================================
def draw_gauge_chart(score):
    """หน้าปัดแสดงระดับความเสี่ยงวิกฤต (Risk Gauge)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "<b>Crisis Risk Score</b><br><span style='font-size:0.8em;color:gray'>Black Swan Indicator</span>",
            'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 40], 'color': "#00cc96"},
                {'range': [40, 70], 'color': "#FFA15A"},
                {'range': [70, 100], 'color': "#EF553B"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
    return fig


# ==========================================================
# SENTIMENT TABLE COLORING
# ==========================================================
def color_sentiment(val):
    """สีตัวอักษรในตารางข่าวตามอารมณ์"""
    color = 'red' if val == 'NEGATIVE' else 'green' if val == 'POSITIVE' else 'orange'
    return f'color: {color}; font-weight: bold'


# ==========================================================
# VIX HISTORY + PREDICTION
# ==========================================================
def draw_vix_history_chart(df, predicted_vix=None, lookback_days=180):
    """กราฟ VIX ย้อนหลัง + จุดทำนายล่วงหน้า 7 วัน + threshold lines"""
    recent = df.tail(lookback_days)

    fig = go.Figure()

    # Historical VIX line
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['VIX'],
        mode='lines', name='VIX (Historical)',
        line=dict(color='#1f77b4', width=2)
    ))

    # 20-day rolling mean (smoother)
    recent_mean = recent['VIX'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent_mean,
        mode='lines', name='VIX 20-day MA',
        line=dict(color='#FFA15A', width=1.5, dash='dot')
    ))

    # Predicted point (7 days ahead)
    if predicted_vix is not None and not pd.isna(predicted_vix):
        future_date = recent.index[-1] + pd.Timedelta(days=7)
        fig.add_trace(go.Scatter(
            x=[recent.index[-1], future_date],
            y=[recent['VIX'].iloc[-1], predicted_vix],
            mode='lines+markers',
            name='AI Forecast (7d)',
            line=dict(color='#EF553B', width=2, dash='dash'),
            marker=dict(size=12, symbol='star')
        ))

    # Crisis threshold lines
    fig.add_hline(y=20, line_dash="dot", line_color="gray", annotation_text="Normal (20)", annotation_position="right")
    fig.add_hline(y=30, line_dash="dash", line_color="orange", annotation_text="Caution (30)", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Crisis (40)", annotation_position="right")

    fig.update_layout(
        title="VIX History & 7-Day AI Forecast",
        xaxis_title="Date", yaxis_title="VIX Level",
        template="plotly_dark", hovermode="x unified",
        height=400, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================
def draw_feature_importance(forecaster):
    """แสดง XGBoost feature importance"""
    if not forecaster.is_trained:
        return None

    importance = forecaster.model.feature_importances_
    features = ['VIX_Lag1', 'VIX_Lag3', 'VIX_Lag7', 'SP500_Return_1D', 'SP500_Return_5D']

    fig = go.Figure(go.Bar(
        x=importance, y=features, orientation='h',
        marker=dict(color=importance, colorscale='Viridis', showscale=False),
        text=[f"{v:.3f}" for v in importance], textposition='outside'
    ))
    fig.update_layout(
        title="Feature Importance (XGBoost)",
        xaxis_title="Importance Score", yaxis_title="",
        template="plotly_dark", height=320,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ==========================================================
# BACKTEST: PRICE + SIGNALS
# ==========================================================
def draw_backtest_chart(df):
    """กราฟราคา S&P 500 + จุดสัญญาณวิกฤต"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines', name='S&P 500 Index',
        line=dict(color='#1f77b4', width=2)
    ))

    if 'Risk_Signal' in df.columns:
        signals = df[df['Risk_Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['Close'],
            mode='markers', name='Black Swan Signal',
            marker=dict(color='red', size=10, symbol='triangle-down',
                        line=dict(color='white', width=1))
        ))

    fig.update_layout(
        title="S&P 500 with Crisis Signals",
        xaxis_title="Date", yaxis_title="Index Price",
        template="plotly_dark", hovermode="x unified",
        height=420, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


# ==========================================================
# BACKTEST: EQUITY CURVE COMPARISON
# ==========================================================
def draw_equity_curve_chart(df):
    """กราฟเปรียบเทียบ cumulative return ของ Strategy vs Buy & Hold"""
    work = df.dropna(subset=['Market_Return', 'Strategy_Return']).copy()
    work['Cum_BuyHold'] = (1 + work['Market_Return']).cumprod()
    work['Cum_Strategy'] = (1 + work['Strategy_Return']).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=work.index, y=work['Cum_BuyHold'],
        mode='lines', name='Buy & Hold',
        line=dict(color='#888888', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=work.index, y=work['Cum_Strategy'],
        mode='lines', name='Black Swan Strategy',
        line=dict(color='#00cc96', width=2)
    ))

    fig.update_layout(
        title="Cumulative Equity Curve (1 = Initial Capital)",
        xaxis_title="Date", yaxis_title="Portfolio Value (×)",
        template="plotly_dark", hovermode="x unified",
        height=380, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


# ==========================================================
# BACKTEST: DRAWDOWN COMPARISON
# ==========================================================
def draw_drawdown_chart(df):
    """กราฟเปรียบเทียบ Drawdown ของ Strategy vs Buy & Hold"""
    work = df.dropna(subset=['Market_Return', 'Strategy_Return']).copy()
    cum_bh = (1 + work['Market_Return']).cumprod()
    cum_st = (1 + work['Strategy_Return']).cumprod()
    dd_bh = ((cum_bh - cum_bh.cummax()) / cum_bh.cummax()) * 100
    dd_st = ((cum_st - cum_st.cummax()) / cum_st.cummax()) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=work.index, y=dd_bh, mode='lines', name='Buy & Hold DD',
        line=dict(color='#888888', width=1.5), fill='tozeroy', fillcolor='rgba(136,136,136,0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=work.index, y=dd_st, mode='lines', name='Strategy DD',
        line=dict(color='#EF553B', width=1.5), fill='tozeroy', fillcolor='rgba(239,85,59,0.3)'
    ))

    fig.update_layout(
        title="Drawdown Comparison (%)",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_dark", hovermode="x unified",
        height=320, legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
    )
    return fig


# ==========================================================
# SENTIMENT DISTRIBUTION DONUT
# ==========================================================
def draw_sentiment_donut(news_df):
    """โดนัทชาร์ตสรุป sentiment ของข่าวล่าสุด"""
    counts = news_df['Sentiment'].value_counts()
    color_map = {'NEGATIVE': '#EF553B', 'NEUTRAL': '#FFA15A', 'POSITIVE': '#00cc96'}
    colors = [color_map.get(s, '#888888') for s in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.55, marker=dict(colors=colors),
        textinfo='label+percent', textfont=dict(size=13)
    ))
    fig.update_layout(
        title="News Sentiment Breakdown",
        template="plotly_dark", height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig
