import plotly.graph_objects as go


def draw_gauge_chart(score):
    """สร้างหน้าปัดแสดงระดับความเสี่ยงวิกฤต (Risk Gauge)"""
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
    return fig


def color_sentiment(val):
    """ฟังก์ชันตกแต่งสีตัวอักษรในตารางข่าวตามอารมณ์"""
    color = 'red' if val == 'NEGATIVE' else 'green' if val == 'POSITIVE' else 'orange'
    return f'color: {color}; font-weight: bold'


def draw_backtest_chart(df):
    """วาดกราฟเส้นราคา S&P 500 พร้อมจุดแจ้งเตือนวิกฤต (Crash Signals)"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='S&P 500 Index',
        line=dict(color='#1f77b4', width=2)
    ))

    if 'Risk_Signal' in df.columns:
        signals = df[df['Risk_Signal'] == 1]
        fig.add_trace(go.Scatter(
            x=signals.index,
            y=signals['Close'],
            mode='markers',
            name='Black Swan Signal',
            marker=dict(
                color='red',
                size=10,
                symbol='triangle-down',
                line=dict(color='white', width=1)
            )
        ))

    fig.update_layout(
        title="Strategy Backtest: S&P 500 vs Black Swan Signals",
        xaxis_title="Date",
        yaxis_title="Index Price",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig
