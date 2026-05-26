"""
Build Report — generate static HTML report (ELI5 / Beginner-friendly version)

วิธีรัน:
    python scripts/build_report.py

Output:
    docs/index.html

แนวคิด: อธิบายเหมือนคุยกับเด็ก 5 ขวบหรือคนที่ไม่รู้เรื่องการเงิน/ML เลย
ใช้การเปรียบเทียบกับสิ่งของในชีวิตประจำวัน (อากาศ, อุณหภูมิ, หมอดู)
"""
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console emoji encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# add parent to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from data.market_crawler import fetch_macro_data, fetch_asset_data
from engine.features import build_features
from engine.regime_detector import classify_market_regime
from engine.ml_predictor import (
    VIXForecaster, walk_forward_validate, compare_models, compute_shap_values
)
from engine.backtester import run_advanced_backtest
from engine.disk_cache import cache_dataframe, cache_model, cache_pickle
from ui.components import (
    draw_vix_history_chart, draw_equity_curve_chart, draw_drawdown_chart,
    draw_backtest_chart, draw_walkforward_chart, draw_model_comparison,
    draw_shap_summary,
)


# ==========================================================
# CONFIG
# ==========================================================
YEARS_LOOKBACK = 5
VIX_THRESHOLD = 30.0
VOL_MULTIPLIER = 1.5
TRANSACTION_COST_BPS = 10
WF_SPLITS = 5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs"
OUTPUT_FILE = OUTPUT_DIR / "index.html"

pio.templates.default = "plotly_dark"


# ==========================================================
# HELPERS
# ==========================================================
def fig_to_html(fig):
    return fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={'displayModeBar': False, 'responsive': True}
    )


def big_status(emoji, title, subtitle, color):
    return f"""
    <div class="big-status" style="background: linear-gradient(135deg, {color}22, {color}11); border: 2px solid {color};">
        <div class="big-emoji">{emoji}</div>
        <div class="big-title">{title}</div>
        <div class="big-subtitle">{subtitle}</div>
    </div>
    """


def fact_card(emoji, label, value, explanation):
    return f"""
    <div class="fact-card">
        <div class="fact-emoji">{emoji}</div>
        <div class="fact-content">
            <div class="fact-label">{label}</div>
            <div class="fact-value">{value}</div>
            <div class="fact-explain">{explanation}</div>
        </div>
    </div>
    """


def speech_bubble(emoji, text):
    return f"""
    <div class="speech-bubble">
        <span class="bubble-emoji">{emoji}</span>
        <span class="bubble-text">{text}</span>
    </div>
    """


# ==========================================================
# ANALYSIS (เหมือนเดิม)
# ==========================================================
def _train_forecaster(macro):
    fc = VIXForecaster("XGBoost")
    fc.train_model(macro)
    return fc


def run_analysis():
    print("📊 [1/9] โหลด macro data (cache 1h)...")
    macro = cache_dataframe(f"macro_{YEARS_LOOKBACK}y",
                            lambda: fetch_macro_data(years=YEARS_LOOKBACK),
                            ttl_hours=1.0)
    if macro is None or macro.empty:
        raise RuntimeError("โหลด macro data ไม่ได้")
    print(f"   ได้ {len(macro)} วัน")

    print("📊 [2/9] วิเคราะห์อารมณ์ตลาด...")
    regime = classify_market_regime(macro)
    print(f"   Regime: {regime}")

    print("🤖 [3/9] โหลด/เทรน XGBoost (cache 24h)...")
    forecaster = cache_model(f"xgb_forecaster_{YEARS_LOOKBACK}y",
                             lambda: _train_forecaster(macro),
                             ttl_hours=24.0)
    predicted_vix = forecaster.predict_vix(macro)
    current_vix = float(macro['VIX'].iloc[-1])
    print(f"   VIX: {current_vix:.2f} -> AI 7d: {predicted_vix:.2f}")

    print("📈 [4/9] Walk-forward validation (cache 24h)...")
    wf_result = cache_pickle(f"wf_xgb_{YEARS_LOOKBACK}y_{WF_SPLITS}f",
                             lambda: walk_forward_validate(macro, "XGBoost",
                                                           task="regression",
                                                           n_splits=WF_SPLITS),
                             ttl_hours=24.0)
    print(f"   R²: {wf_result['mean_score']:.3f}")

    print("⚔️ [5/9] Model comparison (cache 24h)...")
    cmp_reg = cache_pickle(f"cmp_reg_{YEARS_LOOKBACK}y_{WF_SPLITS}f",
                           lambda: compare_models(macro, task="regression",
                                                  n_splits=WF_SPLITS),
                           ttl_hours=24.0)
    winner = cmp_reg.sort_values('Mean Score', ascending=False).iloc[0]
    print(f"   Winner: {winner['Model']} (R²={winner['Mean Score']:.3f})")

    print("🔍 [6/9] SHAP values (cache 24h)...")
    clean_df, feat_cols, _ = build_features(macro, classification=False)
    shap_data = cache_pickle(f"shap_{YEARS_LOOKBACK}y",
                             lambda: compute_shap_values(
                                 forecaster.model, clean_df[feat_cols]),
                             ttl_hours=24.0)
    shap_values, feat_names, X_sample = shap_data

    print("💰 [7/9] Backtest (incl. transaction costs)...")
    bt = macro.copy()
    bt['Vol_20'] = bt['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    bt['Vol_Median_252'] = bt['Vol_20'].rolling(252, min_periods=60).median()
    bt['Risk_Signal'] = (
        (bt['VIX'] > VIX_THRESHOLD) &
        (bt['Vol_20'] > bt['Vol_Median_252'] * VOL_MULTIPLIER)
    ).astype(int)
    bt_metrics = run_advanced_backtest(bt, transaction_cost_bps=TRANSACTION_COST_BPS)

    print("🌍 [8/9] Multi-asset (cache 1h each)...")
    assets = {}
    for name in ["S&P 500", "Bitcoin", "Gold", "Emerging Markets (EEM)"]:
        safe_key = name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "")
        df = cache_dataframe(f"asset_{safe_key}_{YEARS_LOOKBACK}y",
                             lambda n=name: fetch_asset_data(n, YEARS_LOOKBACK),
                             ttl_hours=1.0)
        if df is not None and not df.empty:
            assets[name] = df

    print("📚 [9/9] COVID-2020 case study...")
    covid_data = build_covid_case_study(bt)

    return {
        'macro': macro, 'regime': regime, 'forecaster': forecaster,
        'current_vix': current_vix, 'predicted_vix': predicted_vix,
        'wf_result': wf_result, 'cmp_reg': cmp_reg,
        'shap': (shap_values, feat_names, X_sample),
        'bt': bt, 'bt_metrics': bt_metrics, 'assets': assets,
        'covid': covid_data,
    }


# ==========================================================
# COVID-19 CASE STUDY
# ==========================================================
def build_covid_case_study(bt_data):
    """
    ดู COVID-2020 — ระบบเตือนทันมั้ย?

    Returns dict with:
    - chart_html: Plotly figure ของช่วง COVID
    - first_signal_date: วันแรกที่ระบบเตือน
    - peak_vix_date: วันที่ VIX สูงสุด
    - peak_vix: ค่า VIX สูงสุด
    - drawdown_pct: S&P 500 ตกกี่ %
    - signal_lead_days: ระบบเตือนล่วงหน้ากี่วันก่อน peak
    """
    # ถ้าข้อมูลไม่ครอบคลุม 2020-03 -> skip
    start = pd.Timestamp("2020-02-01")
    end = pd.Timestamp("2020-06-30")

    covid = bt_data.loc[(bt_data.index >= start) & (bt_data.index <= end)].copy()
    if covid.empty or len(covid) < 30:
        return None

    # หา peak VIX
    peak_idx = covid['VIX'].idxmax()
    peak_vix = float(covid['VIX'].max())

    # หา first signal
    signals = covid[covid['Risk_Signal'] == 1]
    first_signal = signals.index[0] if not signals.empty else None

    # หา max drawdown ของ S&P 500
    sp_peak = covid['Close'].iloc[0]
    sp_trough = covid['Close'].min()
    drawdown = ((sp_trough - sp_peak) / sp_peak) * 100

    # ระยะเตือนล่วงหน้า
    lead_days = (peak_idx - first_signal).days if first_signal else None

    # Build chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=covid.index, y=covid['Close'],
        mode='lines', name='S&P 500',
        line=dict(color='#1f77b4', width=2),
        yaxis='y',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>S&P: %{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=covid.index, y=covid['VIX'],
        mode='lines', name='VIX (ความกลัว)',
        line=dict(color='#EF553B', width=2, dash='dot'),
        yaxis='y2',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>VIX: %{y:.1f}<extra></extra>'
    ))

    # Mark signals
    if not signals.empty:
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['Close'],
            mode='markers', name='🚨 ระบบเตือน',
            marker=dict(color='red', size=12, symbol='triangle-down',
                        line=dict(color='white', width=1)),
            yaxis='y',
            hovertemplate='<b>🚨 %{x|%d %b %Y}</b><br>S&P: %{y:,.0f}<extra></extra>'
        ))

    # Annotation: first signal
    if first_signal:
        fig.add_annotation(
            x=first_signal, y=covid.loc[first_signal, 'Close'],
            text=f"🚨 เตือนครั้งแรก<br>{first_signal.strftime('%d %b %Y')}",
            showarrow=True, arrowhead=2, arrowcolor="red",
            ax=-80, ay=-60,
            bgcolor="rgba(239,85,59,0.2)",
            bordercolor="red", borderwidth=1,
            font=dict(color="white"),
        )

    # Annotation: peak VIX
    fig.add_annotation(
        x=peak_idx, y=peak_vix,
        text=f"⛈️ VIX สูงสุด {peak_vix:.0f}<br>{peak_idx.strftime('%d %b %Y')}",
        showarrow=True, arrowhead=2, arrowcolor="orange",
        ax=80, ay=-30, yref='y2',
        bgcolor="rgba(255,161,90,0.2)",
        bordercolor="orange", borderwidth=1,
        font=dict(color="white"),
    )

    fig.update_layout(
        title="🦠 COVID-19 Crash (กุมภาพันธ์ - มิถุนายน 2020)",
        template="plotly_dark", hovermode="x unified", height=500,
        xaxis=dict(title=None),
        yaxis=dict(title="S&P 500", side="left"),
        yaxis2=dict(title="VIX", side="right", overlaying="y",
                    showgrid=False, color="#EF553B"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(t=60, b=40),
    )

    return {
        'fig': fig,
        'first_signal_date': first_signal,
        'peak_vix_date': peak_idx,
        'peak_vix': peak_vix,
        'drawdown_pct': float(drawdown),
        'signal_lead_days': lead_days,
        'n_signals': int(signals.shape[0]),
    }


# ==========================================================
# HTML TEMPLATE — ELI5 (เด็ก 5 ขวบเข้าใจได้)
# ==========================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🚨 วันนี้ตลาดเสี่ยงแค่ไหน?</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Sukhumvit Set", "Noto Sans Thai", sans-serif;
    background: #0e1117; color: #fafafa; line-height: 1.8; font-size: 17px;
  }}
  .container {{ max-width: 980px; margin: 0 auto; padding: 32px 20px; }}

  /* Hero header */
  header {{
    text-align: center; padding: 40px 20px; margin-bottom: 32px;
    background: linear-gradient(135deg, rgba(31,119,180,0.08), rgba(239,85,59,0.08));
    border-radius: 16px;
  }}
  h1 {{ font-size: 2.6rem; margin-bottom: 8px; }}
  .hero-sub {{ color: #aaa; font-size: 1.15rem; }}
  .meta-row {{
    display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;
    margin-top: 20px; font-size: 0.9rem; color: #888;
  }}
  .meta-row a {{ color: #1f77b4; text-decoration: none; }}

  /* Section */
  section {{ margin: 56px 0; }}
  h2 {{
    font-size: 1.8rem; margin-bottom: 16px;
    display: flex; align-items: center; gap: 12px;
  }}
  .section-tagline {{ color: #aaa; font-size: 1.05rem; margin-bottom: 24px; }}
  h3 {{ font-size: 1.3rem; margin: 28px 0 12px; color: #ddd; }}

  /* Big status card */
  .big-status {{
    padding: 40px 32px; text-align: center; border-radius: 16px; margin: 24px 0;
  }}
  .big-emoji {{ font-size: 5rem; line-height: 1; margin-bottom: 16px; }}
  .big-title {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
  .big-subtitle {{ font-size: 1.15rem; color: #ddd; }}

  /* Fact cards */
  .facts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px; margin: 24px 0;
  }}
  .fact-card {{
    background: rgba(255,255,255,0.04);
    padding: 20px; border-radius: 12px;
    display: flex; gap: 16px; align-items: flex-start;
  }}
  .fact-emoji {{ font-size: 2.4rem; line-height: 1; }}
  .fact-content {{ flex: 1; }}
  .fact-label {{ font-size: 0.9rem; color: #aaa; }}
  .fact-value {{ font-size: 1.8rem; font-weight: 700; margin: 4px 0; }}
  .fact-explain {{ font-size: 0.95rem; color: #bbb; }}

  /* Speech bubble */
  .speech-bubble {{
    background: rgba(31,119,180,0.15);
    border-left: 4px solid #1f77b4;
    padding: 18px 22px; border-radius: 8px;
    margin: 20px 0; display: flex; gap: 14px; align-items: flex-start;
  }}
  .bubble-emoji {{ font-size: 1.8rem; line-height: 1; }}
  .bubble-text {{ flex: 1; font-size: 1.05rem; line-height: 1.7; }}
  .bubble-text strong {{ color: #1f77b4; }}

  /* Comparison cards */
  .compare-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px; margin: 24px 0;
  }}
  @media (max-width: 700px) {{
    .compare-grid {{ grid-template-columns: 1fr; }}
  }}
  .compare-card {{
    padding: 24px; border-radius: 12px;
    text-align: center;
  }}
  .compare-card.winner {{
    background: rgba(0,204,150,0.12); border: 2px solid #00cc96;
  }}
  .compare-card.loser {{
    background: rgba(239,85,59,0.08); border: 2px solid rgba(239,85,59,0.4);
  }}
  .compare-emoji {{ font-size: 3rem; }}
  .compare-name {{ font-size: 1.3rem; font-weight: 700; margin: 8px 0; }}
  .compare-num {{ font-size: 2.4rem; font-weight: 700; margin: 8px 0; }}
  .compare-desc {{ color: #ccc; }}

  /* Simple list */
  .simple-list {{
    background: rgba(255,255,255,0.03);
    padding: 20px 28px; border-radius: 12px; margin: 20px 0;
  }}
  .simple-list li {{
    margin: 12px 0; list-style: none; padding-left: 32px; position: relative;
  }}
  .simple-list li::before {{
    content: "👉"; position: absolute; left: 0; top: 0;
  }}

  /* Chart container */
  .chart-wrap {{
    background: rgba(255,255,255,0.02);
    padding: 16px; border-radius: 12px; margin: 20px 0;
  }}

  /* Final summary */
  .final-summary {{
    margin-top: 48px; padding: 32px;
    background: linear-gradient(135deg, rgba(31,119,180,0.15), rgba(0,204,150,0.08));
    border-radius: 16px;
  }}
  .final-summary h2 {{ margin-bottom: 24px; }}
  .summary-item {{ margin: 16px 0; font-size: 1.1rem; line-height: 1.8; }}

  footer {{
    margin-top: 64px; padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.1);
    color: #888; font-size: 0.9rem; text-align: center;
  }}
  footer a {{ color: #1f77b4; text-decoration: none; }}

  /* Toggle for nerds */
  .nerd-mode {{
    margin-top: 32px; padding: 16px;
    background: rgba(255,255,255,0.03); border-radius: 8px;
  }}
  .nerd-mode summary {{
    cursor: pointer; font-weight: 600; color: #aaa;
    list-style: none; padding: 8px 0;
  }}
  .nerd-mode summary::-webkit-details-marker {{ display: none; }}
  .nerd-mode summary::before {{ content: "🤓 "; }}
  .nerd-mode[open] summary::before {{ content: "👇 "; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>🚨 วันนี้ตลาดเสี่ยงแค่ไหน?</h1>
  <p class="hero-sub">
    ระบบเตือนภัยล่วงหน้าก่อนวิกฤตการเงิน — อธิบายง่ายๆ ไม่ต้องเก่งคณิตศาสตร์
  </p>
  <div class="meta-row">
    <span>📅 อัพเดทล่าสุด: <strong>{timestamp}</strong></span>
    <span>📊 ดูข้อมูลย้อนหลัง {years} ปี</span>
    <span><a href="https://github.com/oomNoNe/black-swan-indicator" target="_blank">โค้ดทั้งหมดอยู่ที่นี่ →</a></span>
  </div>
</header>

<!-- ============================================ -->
<!-- SECTION 1: สรุปด่วน — วันนี้เสี่ยงแค่ไหน? -->
<!-- ============================================ -->
<section>
  <h2>🌤️ วันนี้ตลาดเป็นยังไง?</h2>
  <p class="section-tagline">
    เริ่มจากภาพรวมง่ายๆ ก่อน เหมือนเปิดดูพยากรณ์อากาศตอนเช้า
  </p>

  {hero_status}

  <div class="facts-grid">
    {top_facts}
  </div>
</section>

<!-- ============================================ -->
<!-- SECTION 2: VIX = อุณหภูมิตลาด -->
<!-- ============================================ -->
<section>
  <h2>🌡️ VIX คืออะไร? (อุณหภูมิตลาด)</h2>
  <p class="section-tagline">
    เหมือนเทอร์โมมิเตอร์วัดไข้ — แต่วัด "ความกลัว" ของนักลงทุนแทน
  </p>

  {vix_explainer}

  {speech_vix}

  <h3>📈 อุณหภูมิตลาดย้อนหลังครึ่งปี</h3>
  <div class="chart-wrap">{vix_chart}</div>
  <p class="section-tagline">
    💡 <strong>วิธีอ่าน</strong>: เส้นน้ำเงิน = ความกลัวจริง,
    ดาวแดง = AI ทำนายว่าอีก 7 วันความกลัวจะเป็นเท่าไหร่
  </p>
</section>

<!-- ============================================ -->
<!-- SECTION 3: AI หมอดูทำนายอะไร? -->
<!-- ============================================ -->
<section>
  <h2>🔮 เรามีหมอดู AI ที่ทำนายตลาด</h2>
  <p class="section-tagline">
    เราเทรน AI ให้เรียนรู้จากอดีต 5 ปี แล้วให้มันทำนายอนาคต 7 วันข้างหน้า
  </p>

  {forecast_card}

  {speech_forecast}
</section>

<!-- ============================================ -->
<!-- SECTION 4: หมอดู AI แม่นแค่ไหน? -->
<!-- ============================================ -->
<section>
  <h2>🎯 แต่... AI แม่นจริงเหรอ?</h2>
  <p class="section-tagline">
    คำถามสำคัญ — เราเลยทดสอบ AI กับอดีต ดูว่าทำนายผ่านมาถูกบ่อยแค่ไหน
  </p>

  {accuracy_explainer}

  <div class="chart-wrap">{wf_chart}</div>

  {speech_accuracy}
</section>

<!-- ============================================ -->
<!-- SECTION 5: AI 4 ตัวแข่งกัน -->
<!-- ============================================ -->
<section>
  <h2>⚔️ มีหมอดู 4 คนแข่งกัน — ใครเก่งสุด?</h2>
  <p class="section-tagline">
    เราเอา AI 4 แบบมาแข่งกัน แต่ละแบบมีจุดเด่นต่างกัน
  </p>

  <div class="simple-list">
    <ul>
      <li><strong>👶 Naive</strong> = วิธีง่ายที่สุด — "เดาว่าพรุ่งนี้ = วันนี้" (เป็นมาตรฐานเทียบ)</li>
      <li><strong>🌲 XGBoost</strong> = ต้นไม้การตัดสินใจ (ฉลาด ใช้กันมากใน Kaggle)</li>
      <li><strong>💡 LightGBM</strong> = ต้นไม้แบบเร็ว (น้องของ XGBoost)</li>
      <li><strong>📐 Ridge</strong> = สมการคณิตศาสตร์เรียบง่าย (ไม่ฉลาด แต่เสถียร)</li>
      <li><strong>🧠 LSTM</strong> = สมองเลียนแบบมนุษย์ (Deep Learning)</li>
    </ul>
  </div>

  <div class="chart-wrap">{cmp_chart}</div>

  {compare_winner_loser}

  {speech_compare}
</section>

<!-- ============================================ -->
<!-- SECTION 6: AI ดูอะไรเป็นพิเศษ -->
<!-- ============================================ -->
<section>
  <h2>🔍 AI ดูอะไรเป็นพิเศษ?</h2>
  <p class="section-tagline">
    เหมือนเชฟทำต้มยำ — ขอเปิดสูตรว่าใส่วัตถุดิบไหนเยอะที่สุด
  </p>

  {shap_explainer}

  <div class="chart-wrap">{shap_chart}</div>

  {speech_shap}
</section>

<!-- ============================================ -->
<!-- SECTION COVID — Case Study -->
<!-- ============================================ -->
{covid_section}

<!-- ============================================ -->
<!-- SECTION 7: ทดลองเล่นในอดีต -->
<!-- ============================================ -->
<section>
  <h2>💰 ถ้าใช้ระบบนี้จริง จะได้กำไรมั้ย?</h2>
  <p class="section-tagline">
    เราทดลองย้อนหลัง 5 ปี — ถ้ามีเงิน 1 ล้านบาท ใช้กลยุทธ์นี้
    จะได้เท่าไหร่ เทียบกับการซื้อแล้วถือเฉยๆ
  </p>

  {backtest_cards}

  <h3>📈 เงินทุนเติบโตยังไงในช่วง {years} ปี?</h3>
  <div class="chart-wrap">{equity_chart}</div>

  <h3>📉 เคยขาดทุนหนักแค่ไหน?</h3>
  <p class="section-tagline">
    ยิ่งกราฟลงลึก = เจ็บหนัก กลยุทธ์ที่ดี ควรมี "หลุม" ตื้นกว่า
  </p>
  <div class="chart-wrap">{dd_chart}</div>

  <h3>🚨 เคยส่งสัญญาณเตือนเมื่อไหร่บ้าง?</h3>
  <p class="section-tagline">
    จุดสามเหลี่ยมแดง = เวลาที่ระบบบอกว่า "ระวัง! ตลาดอาจจะตก"
  </p>
  <div class="chart-wrap">{signal_chart}</div>

  {speech_backtest}
</section>

<!-- ============================================ -->
<!-- SECTION 8: ตลาดทั่วโลก -->
<!-- ============================================ -->
<section>
  <h2>🌍 ตลาดอื่นเป็นยังไงบ้าง?</h2>
  <p class="section-tagline">
    ดูทุกตลาดทั่วโลกพร้อมกัน — Bitcoin, ทอง, ตลาดเกิดใหม่
    ถ้าทุกอย่างวุ่นวายพร้อมกัน = วิกฤตจริงๆ
  </p>

  <div class="chart-wrap">{multi_asset_chart}</div>

  {speech_multi_asset}
</section>

<!-- ============================================ -->
<!-- FINAL SUMMARY -->
<!-- ============================================ -->
<div class="final-summary">
  <h2>📖 สรุปง่ายๆ</h2>

  <div class="summary-item">
    <strong>🌡️ ตอนนี้ตลาดเสี่ยงแค่ไหน?</strong><br>
    {final_status}
  </div>

  <div class="summary-item">
    <strong>🔮 อีก 7 วันจะเป็นยังไง?</strong><br>
    {final_forecast}
  </div>

  <div class="summary-item">
    <strong>💡 ระบบนี้ใช้ทำอะไรได้?</strong><br>
    ช่วยส่งสัญญาณเตือนล่วงหน้าก่อนวิกฤตการเงิน
    (เช่น ก่อนตลาดตก 30%) เพื่อให้นักลงทุนได้เตรียมตัว
    ไม่ใช่ทำนายแม่นยำ 100% แต่ช่วยลดความเสียหายได้
  </div>

  <div class="summary-item" style="color: #FFA15A;">
    <strong>⚠️ คำเตือน</strong><br>
    นี่เป็น<strong>โปรเจกต์เพื่อการศึกษา</strong>เท่านั้น
    ไม่ใช่คำแนะนำการลงทุน อย่าเอาเงินจริงไปใช้ตามนี้โดยไม่ปรึกษามืออาชีพ
  </div>
</div>

<!-- ============================================ -->
<!-- NERD MODE -->
<!-- ============================================ -->
<details class="nerd-mode">
  <summary>สำหรับคนที่อยากรู้ลึก (เทคนิคจริง)</summary>
  <div style="padding: 16px 0; line-height: 1.8;">
    <h3>📊 ตัวเลขเชิงเทคนิค</h3>
    <ul style="padding-left: 24px;">
      <li><strong>VIX ปัจจุบัน</strong>: {tech_vix}</li>
      <li><strong>AI Forecast (7d)</strong>: {tech_pred} (XGBoost)</li>
      <li><strong>Walk-Forward Mean R²</strong>: {tech_r2} ± {tech_r2_std} ({wf_splits} folds)</li>
      <li><strong>Best Model (regression)</strong>: {tech_best_model}</li>
      <li><strong>Backtest Sharpe</strong>: Strategy {tech_strat_sharpe} vs Buy&Hold {tech_base_sharpe}</li>
      <li><strong>Max Drawdown</strong>: Strategy {tech_strat_mdd}% vs Buy&Hold {tech_base_mdd}%</li>
      <li><strong>Transaction cost</strong>: {tx_cost} bps per turnover</li>
      <li><strong>Features</strong>: 13 (VIX lags, S&P returns, yield curve, gold/oil/DXY momentum)</li>
    </ul>

    <h3>📐 Methodology</h3>
    <ul style="padding-left: 24px;">
      <li>Time-series validation: TimeSeriesSplit, no shuffle (no look-ahead bias)</li>
      <li>FinBERT (ProsusAI) for news sentiment (separate Streamlit app)</li>
      <li>SHAP TreeExplainer for feature attribution</li>
      <li>Multi-asset volatility: realized vol 20d annualized (proxy for non-VIX assets)</li>
      <li>Regime: SMA-50/200 crossover + 20d rolling vol vs historical median</li>
    </ul>

    <p style="margin-top: 16px;">
      Full source code: <a href="https://github.com/oomNoNe/black-swan-indicator" target="_blank">github.com/oomNoNe/black-swan-indicator</a>
    </p>
  </div>
</details>

<footer>
  <p>
    🚨 สร้างโดย <a href="https://github.com/oomNoNe/black-swan-indicator">black-swan-indicator</a>
    · อัพเดทอัตโนมัติทุกสัปดาห์ ·
    <a href="https://github.com/oomNoNe/black-swan-indicator/blob/main/README.th.md">README ภาษาไทย</a>
  </p>
</footer>

</div>
</body>
</html>
"""


# ==========================================================
# BUILD
# ==========================================================
def vix_level_emoji(vix):
    if vix < 15: return "😎", "ตลาดสงบมาก", "#00cc96"
    if vix < 20: return "☀️", "ตลาดสบายๆ", "#00cc96"
    if vix < 25: return "⛅", "เริ่มมีเมฆ", "#FFA15A"
    if vix < 30: return "🌤️", "ต้องระวัง", "#FFA15A"
    if vix < 40: return "⛈️", "อันตราย! ตลาดกลัวมาก", "#EF553B"
    return "🌪️", "วิกฤต! ฟ้าผ่า!", "#EF553B"


def regime_explanation(regime):
    return {
        "Trending Bull": ("📈", "ตลาดขาขึ้น", "หุ้นเฉลี่ยขึ้นมานาน — เหมือนวันแดดออก", "#00cc96"),
        "Panic": ("🔥", "ตลาดวิกฤต", "หุ้นตก + ผันผวนแรง — เหมือนพายุ", "#EF553B"),
        "Ranging": ("⚖️", "ตลาดออกข้าง", "ไม่ขึ้นไม่ลง — เหมือนวันเมฆครึ้ม", "#FFA15A"),
        "Unknown": ("❓", "ข้อมูลไม่พอ", "ยังบอกไม่ได้", "#888"),
    }.get(regime, ("❓", regime, "", "#888"))


def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = run_analysis()

    print("\n📝 กำลังเขียน HTML report (แบบเด็ก 5 ขวบเข้าใจได้)...")

    # ===== Hero status =====
    vix_emoji, vix_label, vix_color = vix_level_emoji(data['current_vix'])
    regime_emoji, regime_label, regime_desc, regime_color = regime_explanation(data['regime'])

    # คำนวณ overall risk level
    if data['current_vix'] < 20 and data['regime'] == "Trending Bull":
        overall_emoji = "☀️"
        overall_title = "ปลอดภัย — วันแดดออก"
        overall_subtitle = "ตลาดสบายๆ ลงทุนตามปกติได้ ความเสี่ยงต่ำ"
        overall_color = "#00cc96"
    elif data['current_vix'] > 30 or data['regime'] == "Panic":
        overall_emoji = "⛈️"
        overall_title = "อันตราย — พายุใกล้มา"
        overall_subtitle = "ตลาดเริ่มกลัว — ระมัดระวังการลงทุน"
        overall_color = "#EF553B"
    else:
        overall_emoji = "⛅"
        overall_title = "ระวัง — เมฆเริ่มก่อตัว"
        overall_subtitle = "ตลาดออกข้าง — ติดตามใกล้ชิด"
        overall_color = "#FFA15A"

    hero_status = big_status(overall_emoji, overall_title, overall_subtitle, overall_color)

    # ===== Top facts =====
    top_facts = (
        fact_card("🌡️", "อุณหภูมิตลาด (VIX)",
                  f"{data['current_vix']:.1f}",
                  f"{vix_emoji} {vix_label}")
        + fact_card("🎭", "อารมณ์ตลาด",
                    regime_label,
                    f"{regime_emoji} {regime_desc}")
        + fact_card("📊", "ดัชนีหุ้นใหญ่ของอเมริกา (S&P 500)",
                    f"{data['macro']['Close'].iloc[-1]:,.0f}",
                    "เป็นตัวแทนตลาดหุ้นโลก")
    )

    # ===== VIX explainer =====
    vix_explainer = """
    <div class="simple-list">
        <ul>
            <li><strong>VIX ต่ำ (0-20)</strong> ☀️ = ตลาดสงบ ทุกคนชิวๆ ลงทุนได้ปกติ</li>
            <li><strong>VIX กลาง (20-30)</strong> ⛅ = เริ่มมีคนกังวล ต้องระวัง</li>
            <li><strong>VIX สูง (30-40)</strong> ⛈️ = ทุกคนกลัว ตลาดผันผวนแรง</li>
            <li><strong>VIX สูงมาก (>40)</strong> 🌪️ = วิกฤต! เคยแตะ 80 ตอน COVID-2020</li>
        </ul>
    </div>
    """

    if data['current_vix'] < 20:
        vix_speech_text = (
            f"VIX ตอนนี้ <strong>{data['current_vix']:.1f}</strong> = "
            f"{vix_emoji} ตลาดสงบมาก เหมือนวันแดดออกไม่มีเมฆ — "
            "นักลงทุนไม่ค่อยกลัวอะไร"
        )
    elif data['current_vix'] < 30:
        vix_speech_text = (
            f"VIX ตอนนี้ <strong>{data['current_vix']:.1f}</strong> = "
            f"{vix_emoji} เริ่มมีเมฆมาก — นักลงทุนเริ่มกังวล แต่ยังไม่ถึงขั้นวิกฤต"
        )
    else:
        vix_speech_text = (
            f"VIX ตอนนี้ <strong>{data['current_vix']:.1f}</strong> = "
            f"{vix_emoji} <strong>ระวัง!</strong> ตลาดกลัวมาก — มักเกิดในช่วงวิกฤตจริง"
        )
    speech_vix = speech_bubble("🧒", vix_speech_text)

    # ===== Forecast =====
    pred_delta = data['predicted_vix'] - data['current_vix']
    if pred_delta > 2:
        forecast_emoji = "📈"
        forecast_msg = "ระวัง! ตลาดอาจกลัวมากขึ้น"
        forecast_color = "#EF553B"
    elif pred_delta > 0:
        forecast_emoji = "↗️"
        forecast_msg = "อาจมีความกังวลเพิ่มเล็กน้อย"
        forecast_color = "#FFA15A"
    else:
        forecast_emoji = "📉"
        forecast_msg = "ดี! สถานการณ์น่าจะคลี่คลาย"
        forecast_color = "#00cc96"

    forecast_card = big_status(
        forecast_emoji,
        f"AI ทำนาย VIX อีก 7 วัน = {data['predicted_vix']:.1f}",
        f"เปลี่ยนแปลง {pred_delta:+.1f} จากวันนี้ — {forecast_msg}",
        forecast_color
    )

    speech_forecast = speech_bubble(
        "🤖",
        f"AI ตัวนี้เรียนรู้รูปแบบของ VIX จาก 5 ปีที่ผ่านมา <strong>1,259 วัน</strong> "
        f"และพยายามทำนายว่าใน 7 วันข้างหน้า VIX จะเป็นเท่าไหร่ "
        f"ครั้งนี้มันบอกว่า <strong>{data['predicted_vix']:.1f}</strong>"
    )

    # ===== Accuracy =====
    r2 = data['wf_result']['mean_score']
    if r2 > 0.3:
        accuracy_msg = "AI แม่นพอใช้ — ทำนายได้ดีกว่าเดาเฉยๆ"
    elif r2 > 0:
        accuracy_msg = "AI แม่นนิดหน่อย — ดีกว่าโยนเหรียญ"
    else:
        accuracy_msg = "AI <strong>ไม่ค่อยแม่น</strong> — VIX เป็นตัวแปรที่ทำนายยากที่สุดในโลกการเงิน แม้แต่ AI ที่ดีที่สุดก็ยังลำบาก"

    accuracy_explainer = f"""
    <div class="simple-list">
        <ul>
            <li>📐 <strong>R² (อาร์-สแควร์)</strong> = คะแนนความแม่นยำ (0 ถึง 1)</li>
            <li>✅ <strong>R² = 1</strong> = แม่นยำ 100% (ทำนายเป๊ะ)</li>
            <li>⚖️ <strong>R² = 0</strong> = แค่เดาค่าเฉลี่ย (ไม่ดีกว่ามนุษย์ทั่วไป)</li>
            <li>❌ <strong>R² ติดลบ</strong> = แย่กว่าเดามั่ว!</li>
        </ul>
    </div>
    <p style="margin: 20px 0; font-size: 1.15rem; text-align: center;">
        ผล AI ตอนนี้: <strong style="font-size: 1.6rem; color: #{('00cc96' if r2 > 0.3 else 'FFA15A' if r2 > 0 else 'EF553B')};">R² = {r2:.3f}</strong>
    </p>
    """

    wf_chart = fig_to_html(draw_walkforward_chart(data['wf_result']['predictions_df']))

    speech_accuracy = speech_bubble("🧒", accuracy_msg)

    # ===== Model Comparison =====
    cmp_fig = draw_model_comparison(data['cmp_reg'])
    cmp_chart = fig_to_html(cmp_fig) if cmp_fig else "<p>ไม่มีข้อมูลเปรียบเทียบ</p>"

    # ✨ คัด best (R² สูงสุด) vs worst (R² ต่ำสุด)
    cmp_sorted = data['cmp_reg'].sort_values('Mean Score', ascending=False).reset_index(drop=True)
    best = cmp_sorted.iloc[0]
    worst = cmp_sorted.iloc[-1]

    compare_winner_loser = f"""
    <div class="compare-grid">
        <div class="compare-card winner">
            <div class="compare-emoji">🥇</div>
            <div class="compare-name">{best['Model']}</div>
            <div class="compare-num">R² = {best['Mean Score']:.3f}</div>
            <div class="compare-desc">ผู้ชนะ — แม่นที่สุดในการทดสอบครั้งนี้</div>
        </div>
        <div class="compare-card loser">
            <div class="compare-emoji">🥉</div>
            <div class="compare-name">{worst['Model']}</div>
            <div class="compare-num">R² = {worst['Mean Score']:.3f}</div>
            <div class="compare-desc">แย่สุดในรอบนี้</div>
        </div>
    </div>
    """

    # ===== Honest insight ขึ้นกับว่า Naive ชนะหรือไม่ =====
    naive_row = data['cmp_reg'][data['cmp_reg']['Model'] == 'Naive']
    if not naive_row.empty:
        naive_score = float(naive_row.iloc[0]['Mean Score'])
        naive_beats_all = best['Model'] == 'Naive'
    else:
        naive_score = None
        naive_beats_all = False

    if naive_beats_all:
        speech_compare = speech_bubble(
            "🤯",
            f"<strong>เซอร์ไพรส์! Naive baseline ชนะทุก AI</strong><br><br>"
            f"'Naive' คือการเดาง่ายๆ ว่า <em>'VIX อีก 7 วัน = VIX วันนี้'</em> "
            f"(R² = {naive_score:.3f}) — ไม่ใช้ AI เลย<br><br>"
            f"แต่กลับชนะ AI ฉลาดทุกตัว (XGBoost, LightGBM, LSTM)<br><br>"
            f"<strong>🎓 บทเรียนสำคัญ</strong>: นี่คือสิ่งที่นักการเงินเรียกว่า "
            f"<strong>'Random Walk Hypothesis'</strong> — ตลาดผันผวนแบบสุ่มมาก "
            f"จน AI ที่ดีที่สุดก็ทำนายไม่ได้ดีกว่า 'พรุ่งนี้ = วันนี้'"
        )
    else:
        speech_compare = speech_bubble(
            "🤓",
            f"<strong>{best['Model']}</strong> ชนะใน task นี้ (R² = {best['Mean Score']:.3f}) "
            f"<br><br>เราใส่ <strong>Naive baseline</strong> (เดาว่า 'VIX อีก 7 วัน = VIX วันนี้') "
            f"เป็นมาตรฐาน — ถ้าโมเดล ML ไม่ชนะ baseline นี้ = ใช้ไม่ได้<br><br>"
            f"บทเรียน: <strong>เริ่มจาก baseline ง่ายๆ เสมอ</strong> ก่อนใช้ AI ใหญ่ๆ"
        )

    # ===== SHAP =====
    shap_values, feat_names, X_sample = data['shap']
    shap_explainer = """
    <div class="simple-list">
        <ul>
            <li>🍜 เหมือนเชฟทำต้มยำ — บางวัตถุดิบสำคัญมาก (พริก ข่า) บางอันสำคัญน้อย (เกลือ)</li>
            <li>🤖 SHAP บอกว่า AI ให้น้ำหนักกับ "วัตถุดิบ" ไหนเยอะที่สุด</li>
            <li>📊 แท่งยิ่งยาว = feature นั้นสำคัญมาก ต่อการตัดสินใจของ AI</li>
        </ul>
    </div>
    """
    if shap_values is not None:
        shap_chart = fig_to_html(draw_shap_summary(shap_values, feat_names, X_sample))
        # หาว่า feature ไหนสำคัญสุด
        importance = np.abs(shap_values).mean(axis=0)
        top_idx = int(np.argmax(importance))
        top_feature = feat_names[top_idx] if feat_names else "unknown"
        speech_shap = speech_bubble(
            "🧒",
            f"AI ของเราดู <strong>{top_feature}</strong> มากที่สุด — "
            "นี่คือ 'วัตถุดิบหลัก' ที่ AI ใช้ตัดสินใจ"
        )
    else:
        shap_chart = "<p>ไม่มี SHAP data</p>"
        speech_shap = ""

    # ===== Backtest =====
    strat = data['bt_metrics']["Black_Swan_Strategy"]
    base = data['bt_metrics']["Baseline_Buy_Hold"]
    ts = data['bt_metrics']["Trading_Stats"]
    sharpe_diff = strat['Sharpe Ratio'] - base['Sharpe Ratio']

    # คำนวณ final return จาก equity curve
    bt_clean = data['bt'].dropna(subset=['Market_Return', 'Strategy_Return'])
    strat_return_pct = ((1 + bt_clean['Strategy_Return']).prod() - 1) * 100
    base_return_pct = ((1 + bt_clean['Market_Return']).prod() - 1) * 100

    initial = 1_000_000
    strat_final = initial * (1 + strat_return_pct / 100)
    base_final = initial * (1 + base_return_pct / 100)

    backtest_cards = f"""
    <div class="compare-grid">
        <div class="compare-card {'winner' if strat_return_pct > base_return_pct else 'loser'}">
            <div class="compare-emoji">🛡️</div>
            <div class="compare-name">ใช้กลยุทธ์ของเรา</div>
            <div class="compare-num">{strat_final:,.0f} บาท</div>
            <div class="compare-desc">
                เริ่มต้น 1 ล้าน → {strat_return_pct:+.1f}%<br>
                ขาดทุนสูงสุด: <strong>{strat['Max Drawdown (%)']:.1f}%</strong>
            </div>
        </div>
        <div class="compare-card {'winner' if base_return_pct > strat_return_pct else 'loser'}">
            <div class="compare-emoji">📊</div>
            <div class="compare-name">ซื้อแล้วถือเฉยๆ</div>
            <div class="compare-num">{base_final:,.0f} บาท</div>
            <div class="compare-desc">
                เริ่มต้น 1 ล้าน → {base_return_pct:+.1f}%<br>
                ขาดทุนสูงสุด: <strong>{base['Max Drawdown (%)']:.1f}%</strong>
            </div>
        </div>
    </div>
    <p class="section-tagline" style="text-align: center;">
        💰 รวมค่าธรรมเนียมการซื้อขายแล้ว ({ts['Number of Trades']} ครั้ง) —
        ใกล้เคียงความจริง
    </p>
    """

    equity_chart = fig_to_html(draw_equity_curve_chart(data['bt']))
    dd_chart = fig_to_html(draw_drawdown_chart(data['bt']))
    signal_chart = fig_to_html(draw_backtest_chart(data['bt']))

    if strat_return_pct > base_return_pct:
        bt_msg = (
            f"🎉 กลยุทธ์ของเรา <strong>ชนะ</strong> การซื้อแล้วถือเฉยๆ! "
            f"ได้เงินมากกว่า <strong>{strat_final - base_final:,.0f} บาท</strong> "
            f"และ <strong>ขาดทุนน้อยกว่า</strong>ในช่วงตลาดตก"
        )
    else:
        bt_msg = (
            f"❌ กลยุทธ์ของเรา <strong>แพ้</strong> การซื้อแล้วถือเฉยๆ ในช่วงนี้ "
            f"เพราะ <strong>{YEARS_LOOKBACK} ปีที่ผ่านมาตลาดส่วนใหญ่ขาขึ้น</strong> "
            f"ระบบป้องกันเลย sacrifice กำไรในช่วงดีไป — เหมือนคนใส่หมวกกันน็อคเดินบนทางที่ปลอดภัย "
            f"<br><br>กลยุทธ์นี้จะแสดงคุณค่าจริง <strong>ตอนวิกฤต</strong> เช่น COVID-2020 หรือ 2008"
        )
    speech_backtest = speech_bubble("🧒", bt_msg)

    # ===== COVID Case Study =====
    covid = data.get('covid')
    if covid:
        lead_text = (f"<strong>{covid['signal_lead_days']} วันก่อน</strong>"
                     if covid['signal_lead_days'] and covid['signal_lead_days'] > 0
                     else "ในช่วงเดียวกับ")
        first_sig_text = (covid['first_signal_date'].strftime('%d %b %Y')
                          if covid['first_signal_date'] else "ไม่ได้เตือน")
        peak_text = covid['peak_vix_date'].strftime('%d %b %Y')

        covid_section = f"""
<section>
  <h2>🦠 Case Study: COVID-19 Crash 2020</h2>
  <p class="section-tagline">
    ลองดูตัวอย่างจริง — ระบบของเราเตือนทันเหตุการณ์วิกฤตที่ใหญ่ที่สุดในรอบ 10 ปีมั้ย?
  </p>

  <div class="facts-grid">
    {fact_card("🚨", "ระบบเตือนครั้งแรก", first_sig_text,
               f"{lead_text}จุดสูงสุดของ VIX")}
    {fact_card("⛈️", "VIX แตะจุดสูงสุด", f"{covid['peak_vix']:.0f}",
               f"เมื่อวันที่ {peak_text}")}
    {fact_card("📉", "S&P 500 ตก", f"{covid['drawdown_pct']:.1f}%",
               "จาก peak ถึง trough ในช่วงนี้")}
    {fact_card("🔔", "จำนวนสัญญาณเตือน", f"{covid['n_signals']} ครั้ง",
               "ตลอดช่วง ก.พ. - มิ.ย. 2020")}
  </div>

  <div class="chart-wrap">{fig_to_html(covid['fig'])}</div>

  {speech_bubble("🎓",
    f"<strong>นี่คือสิ่งที่ระบบควรทำได้</strong> — เตือนล่วงหน้าก่อนวิกฤตจริง "
    f"ในกรณี COVID ระบบส่งสัญญาณวันที่ <strong>{first_sig_text}</strong> "
    f"ซึ่ง {lead_text}จุดที่ตลาดผันผวนสูงสุด<br><br>"
    "ถ้านักลงทุนเชื่อสัญญาณ → ออกจากตลาดเป็นเงินสด → หลีกเลี่ยงการขาดทุน "
    f"<strong>{abs(covid['drawdown_pct']):.0f}%</strong> ในเวลา 1 เดือน<br><br>"
    "<em>แม้ AI ทำนาย VIX ไม่แม่น (Issue ที่เราพบ) — แต่ระบบ rule-based "
    "(VIX > 30 + vol spike) ทำงานได้จริงในวิกฤต</em>"
  )}
</section>
"""
    else:
        covid_section = ""

    # ===== Multi-asset =====
    if data['assets']:
        fig = go.Figure()
        for name, df in data['assets'].items():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['VIX'], mode='lines',
                name=name, line=dict(width=1.5)
            ))
        fig.update_layout(
            title="ความวุ่นวายในตลาดทั่วโลก",
            xaxis_title="เวลา", yaxis_title="ระดับความวุ่นวาย",
            template="plotly_dark", height=420, hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        multi_asset_chart = fig_to_html(fig)
    else:
        multi_asset_chart = "<p>โหลดข้อมูลไม่ได้</p>"

    speech_multi_asset = speech_bubble(
        "🌍",
        "Bitcoin มักผันผวนสูงกว่าหุ้นปกติ 3-5 เท่า ในขณะที่ <strong>ทอง</strong> เป็น 'ที่หลบภัย' "
        "เวลามีวิกฤต — คนแห่ซื้อทำให้ราคาขึ้น "
        "<br><br>ถ้าเห็น<strong>ทุกตลาดวุ่นวายพร้อมกัน</strong> = อันตรายระดับโลก (เช่น COVID-2020)"
    )

    # ===== Final summary =====
    final_status = f"{overall_emoji} <strong>{overall_title}</strong> — {overall_subtitle}"

    if pred_delta > 2:
        final_forecast = f"⚠️ AI ทำนายว่าตลาดจะ <strong>กลัวมากขึ้น</strong> (VIX เพิ่ม {pred_delta:.1f})"
    elif pred_delta > 0:
        final_forecast = f"↗️ ตลาดอาจ <strong>กังวลขึ้นเล็กน้อย</strong> (VIX เพิ่ม {pred_delta:.1f})"
    else:
        final_forecast = f"✅ ตลาดน่าจะ <strong>คลี่คลาย</strong> (VIX ลด {abs(pred_delta):.1f})"

    # ===== Render =====
    html = HTML_TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        years=YEARS_LOOKBACK,
        hero_status=hero_status,
        top_facts=top_facts,
        vix_explainer=vix_explainer,
        speech_vix=speech_vix,
        vix_chart=fig_to_html(draw_vix_history_chart(data['macro'], predicted_vix=data['predicted_vix'])),
        forecast_card=forecast_card,
        speech_forecast=speech_forecast,
        accuracy_explainer=accuracy_explainer,
        wf_chart=wf_chart,
        speech_accuracy=speech_accuracy,
        cmp_chart=cmp_chart,
        compare_winner_loser=compare_winner_loser,
        speech_compare=speech_compare,
        shap_explainer=shap_explainer,
        shap_chart=shap_chart,
        speech_shap=speech_shap,
        backtest_cards=backtest_cards,
        equity_chart=equity_chart,
        dd_chart=dd_chart,
        signal_chart=signal_chart,
        speech_backtest=speech_backtest,
        covid_section=covid_section,
        multi_asset_chart=multi_asset_chart,
        speech_multi_asset=speech_multi_asset,
        final_status=final_status,
        final_forecast=final_forecast,
        # Tech section
        tech_vix=f"{data['current_vix']:.2f}",
        tech_pred=f"{data['predicted_vix']:.2f}",
        tech_r2=f"{data['wf_result']['mean_score']:.3f}",
        tech_r2_std=f"{data['wf_result']['std_score']:.3f}",
        tech_best_model=best['Model'],
        tech_strat_sharpe=f"{strat['Sharpe Ratio']:.2f}",
        tech_base_sharpe=f"{base['Sharpe Ratio']:.2f}",
        tech_strat_mdd=f"{strat['Max Drawdown (%)']:.2f}",
        tech_base_mdd=f"{base['Max Drawdown (%)']:.2f}",
        wf_splits=WF_SPLITS, tx_cost=TRANSACTION_COST_BPS,
    )

    OUTPUT_FILE.write_text(html, encoding='utf-8')
    print(f"\n✅ Report saved: {OUTPUT_FILE}")
    print(f"📦 Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    build()
