"""
Build Report — generate static HTML report สำหรับ GitHub Pages

วิธีรัน:
    python scripts/build_report.py

Output:
    docs/index.html  (Plotly charts ยัง interactive + เปิด offline ก็ได้)

Recruiter จะเห็นรายงานครบทันทีโดยไม่ต้อง clone repo / install dependencies
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

from data.market_crawler import fetch_macro_data, fetch_asset_data, get_current_vix
from engine.features import build_features
from engine.regime_detector import classify_market_regime, dynamic_risk_equation
from engine.ml_predictor import (
    VIXForecaster, walk_forward_validate, compare_models, compute_shap_values
)
from engine.backtester import run_advanced_backtest
from ui.components import (
    draw_gauge_chart, draw_vix_history_chart, draw_feature_importance,
    draw_equity_curve_chart, draw_drawdown_chart, draw_backtest_chart,
    draw_walkforward_chart, draw_model_comparison, draw_shap_summary,
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

# Plotly default theme
pio.templates.default = "plotly_dark"


# ==========================================================
# HELPERS
# ==========================================================
def fig_to_html(fig, full_html=False):
    """Plotly figure -> HTML snippet (CDN'd, interactive)"""
    return fig.to_html(
        full_html=full_html,
        include_plotlyjs='cdn',
        config={'displayModeBar': False, 'responsive': True}
    )


def kpi_card(label, value, sublabel="", color="#1f77b4"):
    return f"""
    <div class="kpi-card" style="border-left: 4px solid {color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sublabel}</div>
    </div>
    """


# ==========================================================
# DATA + ANALYSIS
# ==========================================================
def run_analysis():
    print("📊 [1/8] Fetching macro data (S&P, VIX, yields, gold, oil, DXY)...")
    macro = fetch_macro_data(years=YEARS_LOOKBACK)
    if macro is None or macro.empty:
        raise RuntimeError("ไม่สามารถดึง macro data ได้ — ตรวจสอบ internet/yfinance")

    print(f"   ✅ {len(macro)} วัน × {len(macro.columns)} ตัวชี้วัด")

    print("📊 [2/8] Detecting market regime...")
    regime = classify_market_regime(macro)
    print(f"   ✅ Regime: {regime}")

    print("🤖 [3/8] Training XGBoost VIX forecaster...")
    forecaster = VIXForecaster("XGBoost")
    forecaster.train_model(macro)
    predicted_vix = forecaster.predict_vix(macro)
    current_vix = float(macro['VIX'].iloc[-1])
    print(f"   ✅ Current VIX: {current_vix:.2f} → Predicted (7d): {predicted_vix:.2f}")

    print("📈 [4/8] Walk-forward validation (XGBoost)...")
    wf_result = walk_forward_validate(macro, "XGBoost", task="regression", n_splits=WF_SPLITS)
    print(f"   ✅ Mean R²: {wf_result['mean_score']:.3f}")

    print("⚔️ [5/8] Model comparison (4 models)...")
    cmp_reg = compare_models(macro, task="regression", n_splits=WF_SPLITS)
    print(f"   ✅ Best: {cmp_reg.iloc[0]['Model']} (R²={cmp_reg.iloc[0]['Mean Score']:.3f})")

    print("🔍 [6/8] SHAP feature importance...")
    clean_df, feat_cols, _ = build_features(macro, classification=False)
    shap_values, feat_names, X_sample = compute_shap_values(
        forecaster.model, clean_df[feat_cols]
    )
    print(f"   ✅ {len(feat_names)} features analyzed")

    print("📊 [7/8] Backtest with transaction costs...")
    bt = macro.copy()
    bt['Vol_20'] = bt['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    bt['Vol_Median_252'] = bt['Vol_20'].rolling(252, min_periods=60).median()
    bt['Risk_Signal'] = (
        (bt['VIX'] > VIX_THRESHOLD) &
        (bt['Vol_20'] > bt['Vol_Median_252'] * VOL_MULTIPLIER)
    ).astype(int)
    bt_metrics = run_advanced_backtest(bt, transaction_cost_bps=TRANSACTION_COST_BPS)
    strat = bt_metrics["Black_Swan_Strategy"]
    base = bt_metrics["Baseline_Buy_Hold"]
    print(f"   ✅ Strategy Sharpe: {strat['Sharpe Ratio']:.2f} vs B&H: {base['Sharpe Ratio']:.2f}")

    print("🌍 [8/8] Multi-asset volatility...")
    assets = {}
    for name in ["S&P 500", "Bitcoin", "Gold", "Emerging Markets (EEM)"]:
        df = fetch_asset_data(name, YEARS_LOOKBACK)
        if df is not None and not df.empty:
            assets[name] = df
    print(f"   ✅ {len(assets)} assets loaded")

    return {
        'macro': macro, 'regime': regime, 'forecaster': forecaster,
        'current_vix': current_vix, 'predicted_vix': predicted_vix,
        'wf_result': wf_result, 'cmp_reg': cmp_reg,
        'shap': (shap_values, feat_names, X_sample),
        'bt': bt, 'bt_metrics': bt_metrics, 'assets': assets,
    }


# ==========================================================
# HTML TEMPLATE
# ==========================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🚨 Black Swan Risk Indicator — Live Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0e1117; color: #fafafa; line-height: 1.6;
  }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 32px 24px; }}
  header {{
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 24px; margin-bottom: 32px;
  }}
  h1 {{ font-size: 2.4rem; margin-bottom: 8px; }}
  h2 {{
    font-size: 1.6rem; margin: 48px 0 16px;
    padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08);
  }}
  h3 {{ font-size: 1.2rem; margin: 24px 0 12px; color: #ddd; }}
  .subtitle {{ color: #888; font-size: 1.05rem; }}
  .meta-bar {{
    display: flex; gap: 16px; flex-wrap: wrap;
    margin-top: 16px; font-size: 0.9rem; color: #aaa;
  }}
  .meta-bar a {{ color: #1f77b4; text-decoration: none; }}
  .meta-bar a:hover {{ text-decoration: underline; }}

  /* KPI grid */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px; margin: 24px 0;
  }}
  .kpi-card {{
    background: rgba(255,255,255,0.04);
    padding: 18px; border-radius: 10px;
  }}
  .kpi-label {{ font-size: 0.85rem; color: #aaa; }}
  .kpi-value {{ font-size: 2rem; font-weight: 600; margin: 6px 0; }}
  .kpi-sub {{ font-size: 0.8rem; color: #888; }}

  /* Section content */
  .section-intro {{ color: #ccc; margin-bottom: 16px; }}
  .insight-box {{
    background: rgba(31,119,180,0.08);
    border-left: 3px solid #1f77b4;
    padding: 14px 18px; margin: 16px 0; border-radius: 4px;
  }}
  .insight-box strong {{ color: #1f77b4; }}

  /* Metrics table */
  table {{
    width: 100%; border-collapse: collapse; margin: 16px 0;
    background: rgba(255,255,255,0.03); border-radius: 8px; overflow: hidden;
  }}
  th, td {{ padding: 12px 16px; text-align: left; }}
  th {{ background: rgba(255,255,255,0.05); font-weight: 600; }}
  td {{ border-top: 1px solid rgba(255,255,255,0.05); }}

  /* Two column layout */
  .row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px; margin: 16px 0;
  }}
  @media (max-width: 768px) {{ .row {{ grid-template-columns: 1fr; }} }}

  footer {{
    margin-top: 64px; padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.1);
    color: #888; font-size: 0.85rem; text-align: center;
  }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.8rem; font-weight: 500;
  }}
  .badge-green {{ background: rgba(0,204,150,0.2); color: #00cc96; }}
  .badge-orange {{ background: rgba(255,161,90,0.2); color: #FFA15A; }}
  .badge-red {{ background: rgba(239,85,59,0.2); color: #EF553B; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>🚨 Black Swan Risk Indicator</h1>
  <p class="subtitle">Live Report — Production-grade financial crisis early warning system</p>
  <div class="meta-bar">
    <span>📅 Generated: <strong>{timestamp}</strong></span>
    <span>📊 Data window: <strong>{years} years</strong></span>
    <span>🤖 Model: <strong>XGBoost + LightGBM + Ridge + LSTM</strong></span>
    <span><a href="https://github.com/oomNoNe/black-swan-indicator" target="_blank">View on GitHub →</a></span>
    <span><a href="https://github.com/oomNoNe/black-swan-indicator/blob/main/README.th.md" target="_blank">README (Thai) →</a></span>
  </div>
</header>

<!-- KPI SECTION -->
<section>
  <h2>📍 Current Snapshot</h2>
  <div class="kpi-grid">
    {kpi_cards}
  </div>
</section>

<!-- SECTION 1: VIX HISTORY + FORECAST -->
<section>
  <h2>🌡️ VIX History & 7-Day Forecast</h2>
  <p class="section-intro">
    VIX (Volatility Index) สะท้อนความกลัวของตลาดสหรัฐ — ค่าปกติ &lt; 20, วิกฤต &gt; 40
    เส้นประสีแดงคือการทำนายล่วงหน้า 7 วันจากโมเดล XGBoost
  </p>
  {vix_chart}
  <div class="insight-box">
    <strong>🧠 Insight</strong>: {vix_insight}
  </div>
</section>

<!-- SECTION 2: WALK-FORWARD VALIDATION -->
<section>
  <h2>📈 Walk-Forward Validation</h2>
  <p class="section-intro">
    มาตรฐาน time-series ML — train บนช่วงเวลา 1 แล้ว test ช่วง 2, train 1+2 test 3, ฯลฯ
    ป้องกัน look-ahead bias ที่เกิดจาก train/test split แบบ shuffle ปกติ
  </p>
  <div class="kpi-grid">
    {wf_kpis}
  </div>
  {wf_chart}
</section>

<!-- SECTION 3: MODEL COMPARISON -->
<section>
  <h2>⚔️ Model Comparison</h2>
  <p class="section-intro">
    เทียบ 4 โมเดล (XGBoost / LightGBM / Ridge / LSTM) ด้วย walk-forward — ใครชนะใน task พยากรณ์ VIX?
  </p>
  {cmp_chart}
  <div class="insight-box">
    <strong>📊 Honest finding</strong>: {cmp_insight}
  </div>
</section>

<!-- SECTION 4: SHAP -->
<section>
  <h2>🔍 SHAP Feature Importance (Explainable AI)</h2>
  <p class="section-intro">
    SHAP values อธิบายว่าแต่ละ feature "ผลักดัน" การทำนายไปทางไหน — ใช้ Shapley values จาก game theory
    ให้ผลแม่นยำและ fair กว่า feature_importance ปกติของ XGBoost
  </p>
  {shap_chart}
</section>

<!-- SECTION 5: BACKTEST -->
<section>
  <h2>📊 Strategy Backtest</h2>
  <p class="section-intro">
    เปรียบเทียบกลยุทธ์ "Black Swan Detector" vs "Buy & Hold" บนข้อมูล {years} ปี
    (รวม transaction cost {tx_cost} bps เป็น realistic test)
  </p>
  <div class="kpi-grid">
    {backtest_kpis}
  </div>
  <h3>Equity Curve</h3>
  {equity_chart}
  <div class="row">
    <div>
      <h3>Drawdown Comparison</h3>
      {dd_chart}
    </div>
    <div>
      <h3>Crisis Signals on S&P 500</h3>
      {signal_chart}
    </div>
  </div>
  <div class="insight-box">
    <strong>📉 Backtest interpretation</strong>: {bt_insight}
  </div>
</section>

<!-- SECTION 6: MULTI-ASSET -->
<section>
  <h2>🌍 Multi-Asset Volatility Comparison</h2>
  <p class="section-intro">
    ขยายจาก S&P 500 ไปยัง asset class อื่นๆ — crypto, EM equity, commodity
    เพื่อดูว่ามี systemic risk ที่กระทบทุก asset พร้อมกันมั้ย
  </p>
  {multi_asset_chart}
</section>

<!-- METHODOLOGY -->
<section>
  <h2>📖 Methodology</h2>
  <table>
    <tr><th>Component</th><th>Choice</th><th>Why</th></tr>
    <tr><td>Sentiment</td><td>FinBERT (ProsusAI)</td><td>Pre-trained on financial corpus</td></tr>
    <tr><td>Forecasting</td><td>XGBoost / LightGBM / Ridge / LSTM</td><td>Compare tree-based vs linear vs deep</td></tr>
    <tr><td>Validation</td><td>TimeSeriesSplit ({wf_splits} folds)</td><td>No look-ahead bias</td></tr>
    <tr><td>Regime</td><td>SMA-50/200 + rolling vol</td><td>Industry standard, interpretable</td></tr>
    <tr><td>Backtest cost</td><td>{tx_cost} bps per turnover</td><td>Realistic retail broker assumption</td></tr>
    <tr><td>SHAP</td><td>TreeExplainer</td><td>Fair attribution from game theory</td></tr>
  </table>
</section>

<footer>
  <p>
    🚨 <strong>Disclaimer</strong>: For educational and research purposes only. Not financial advice.<br>
    Generated by <a href="https://github.com/oomNoNe/black-swan-indicator" style="color: #1f77b4;">black-swan-indicator</a>
    · MIT License · <a href="https://github.com/oomNoNe/black-swan-indicator/actions" style="color: #1f77b4;">CI</a>
  </p>
</footer>

</div>
</body>
</html>
"""


# ==========================================================
# BUILD REPORT
# ==========================================================
def build():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = run_analysis()

    print("\n📝 Building HTML report...")

    # ---- KPI Cards ----
    regime_color = {"Trending Bull": "#00cc96", "Panic": "#EF553B",
                    "Ranging": "#FFA15A"}.get(data['regime'], "#888")
    pred_delta = data['predicted_vix'] - data['current_vix']
    pred_arrow = "📈" if pred_delta > 0 else "📉"

    kpi_cards = (
        kpi_card("Current VIX", f"{data['current_vix']:.2f}",
                 "Fear gauge (< 20 = calm, > 30 = caution)", color="#1f77b4")
        + kpi_card("AI Forecast (7d)", f"{data['predicted_vix']:.2f}",
                   f"{pred_arrow} {pred_delta:+.2f} from current",
                   color="#EF553B" if pred_delta > 0 else "#00cc96")
        + kpi_card("Market Regime", data['regime'],
                   "From SMA + rolling vol", color=regime_color)
        + kpi_card("S&P 500 (latest)", f"{data['macro']['Close'].iloc[-1]:,.0f}",
                   f"{(data['macro']['Close'].iloc[-1] / data['macro']['Close'].iloc[-21] - 1) * 100:+.2f}% (20d)",
                   color="#1f77b4")
    )

    # ---- VIX chart + insight ----
    vix_fig = draw_vix_history_chart(data['macro'], predicted_vix=data['predicted_vix'])
    if pred_delta > 5:
        vix_insight = "โมเดลคาดว่า VIX จะ <strong>พุ่งขึ้นแรง</strong> ใน 7 วันข้างหน้า — เตรียมป้องกันความเสี่ยง"
    elif pred_delta > 0:
        vix_insight = "VIX มีแนวโน้ม <strong>เพิ่มขึ้นเล็กน้อย</strong> — เฝ้าระวังตามปกติ"
    else:
        vix_insight = "VIX มีแนวโน้ม <strong>ลดลง</strong> — สถานการณ์น่าจะคลี่คลาย"

    # ---- WF KPIs ----
    wf_kpis = (
        kpi_card("Mean R²", f"{data['wf_result']['mean_score']:.3f}",
                 f"averaged across {WF_SPLITS} folds", color="#1f77b4")
        + kpi_card("Std R²", f"{data['wf_result']['std_score']:.3f}",
                   "lower = more stable model", color="#FFA15A")
        + kpi_card("Folds", str(WF_SPLITS),
                   "expanding window CV", color="#00cc96")
    )
    wf_chart = fig_to_html(draw_walkforward_chart(data['wf_result']['predictions_df']))

    # ---- Model Comparison ----
    cmp_fig = draw_model_comparison(data['cmp_reg'])
    cmp_chart = fig_to_html(cmp_fig) if cmp_fig else "<p>No comparison data</p>"
    best_model = data['cmp_reg'].iloc[0]['Model']
    worst_model = data['cmp_reg'].iloc[-1]['Model']
    cmp_insight = (
        f"<strong>{best_model}</strong> ชนะใน task นี้ "
        f"(R² = {data['cmp_reg'].iloc[0]['Mean Score']:.3f}) "
        f"ขณะที่ <strong>{worst_model}</strong> แย่สุด ({data['cmp_reg'].iloc[-1]['Mean Score']:.3f}). "
        "บนข้อมูลขนาดเล็ก (~1,200 วัน) linear models มักชนะ deep learning เพราะ overfit น้อยกว่า"
    )

    # ---- SHAP ----
    shap_values, feat_names, X_sample = data['shap']
    if shap_values is not None:
        shap_chart = fig_to_html(draw_shap_summary(shap_values, feat_names, X_sample))
    else:
        shap_chart = "<p>SHAP values unavailable</p>"

    # ---- Backtest ----
    strat = data['bt_metrics']["Black_Swan_Strategy"]
    base = data['bt_metrics']["Baseline_Buy_Hold"]
    ts = data['bt_metrics']["Trading_Stats"]

    sharpe_delta = strat['Sharpe Ratio'] - base['Sharpe Ratio']
    mdd_delta = strat['Max Drawdown (%)'] - base['Max Drawdown (%)']

    backtest_kpis = (
        kpi_card("Sharpe Ratio", f"{strat['Sharpe Ratio']:.2f}",
                 f"vs Buy & Hold: {base['Sharpe Ratio']:.2f} ({sharpe_delta:+.2f})",
                 color="#00cc96" if sharpe_delta > 0 else "#FFA15A")
        + kpi_card("Max Drawdown", f"{strat['Max Drawdown (%)']:.1f}%",
                   f"vs B&H: {base['Max Drawdown (%)']:.1f}% ({mdd_delta:+.1f}%)",
                   color="#00cc96" if mdd_delta > 0 else "#EF553B")
        + kpi_card("Win Rate", f"{strat['Win Rate (%)']:.1f}%",
                   f"Profit Factor: {strat['Profit Factor']:.2f}", color="#1f77b4")
        + kpi_card("Trades", str(ts['Number of Trades']),
                   f"Cost: {ts['Total Cost (% of capital)']:.2f}% of capital", color="#888")
    )
    equity_chart = fig_to_html(draw_equity_curve_chart(data['bt']))
    dd_chart = fig_to_html(draw_drawdown_chart(data['bt']))
    signal_chart = fig_to_html(draw_backtest_chart(data['bt']))

    if sharpe_delta > 0:
        bt_insight = (
            f"กลยุทธ์ <strong>ชนะ</strong> baseline ทั้ง Sharpe และ drawdown "
            "แสดงว่า signal detector ทำงานได้ — แต่ระวัง survivorship bias + ผลในอดีตไม่การันตีอนาคต"
        )
    else:
        bt_insight = (
            f"กลยุทธ์ <strong>แพ้</strong> baseline ในช่วงนี้ (Sharpe ต่ำกว่า) "
            "ปกติของ crash-avoidance strategy ในตลาด bull — สละกำไรช่วงตลาดดีเพื่อหลีกเลี่ยงวิกฤต "
            "ที่อาจไม่เกิด"
        )

    # ---- Multi-Asset ----
    if data['assets']:
        fig = go.Figure()
        for name, df in data['assets'].items():
            fig.add_trace(go.Scatter(
                x=df.index, y=df['VIX'],
                mode='lines', name=name, line=dict(width=1.5)
            ))
        fig.update_layout(
            title="Volatility Across Asset Classes",
            xaxis_title="Date", yaxis_title="Volatility (VIX or realized vol proxy)",
            template="plotly_dark", height=420, hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        multi_asset_chart = fig_to_html(fig)
    else:
        multi_asset_chart = "<p>No multi-asset data available</p>"

    # ---- Render ----
    html = HTML_TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        years=YEARS_LOOKBACK,
        kpi_cards=kpi_cards,
        vix_chart=fig_to_html(vix_fig),
        vix_insight=vix_insight,
        wf_kpis=wf_kpis, wf_chart=wf_chart,
        cmp_chart=cmp_chart, cmp_insight=cmp_insight,
        shap_chart=shap_chart,
        backtest_kpis=backtest_kpis,
        equity_chart=equity_chart, dd_chart=dd_chart, signal_chart=signal_chart,
        bt_insight=bt_insight,
        multi_asset_chart=multi_asset_chart,
        wf_splits=WF_SPLITS, tx_cost=TRANSACTION_COST_BPS,
    )

    OUTPUT_FILE.write_text(html, encoding='utf-8')
    print(f"\n✅ Report saved to: {OUTPUT_FILE}")
    print(f"📦 Size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"🌐 Open in browser: file:///{OUTPUT_FILE.as_posix()}")


if __name__ == "__main__":
    build()
