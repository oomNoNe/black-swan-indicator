"""
Build Report — generate static HTML reports in Thai + English

วิธีรัน:
    python scripts/build_report.py            # builds both
    python scripts/build_report.py --lang en  # English only
    python scripts/build_report.py --lang th  # Thai only

Output:
    docs/index.html      (Thai, default landing)
    docs/index.en.html   (English)

แนวคิด: เนื้อหาเขียนสำหรับ "เด็ก ม.3 ที่อยากรู้เรื่องการเงิน" —
ใช้ศัพท์จริง (VIX, R², Sharpe) แต่มีคำอธิบายสั้นๆ ทุกครั้ง
ไม่ผิดทาง 5 ขวบ (กำกวมเกินไป) แต่ก็ไม่ jargon จัดจน outsider ไม่เข้าใจ
"""
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

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
    draw_shap_summary, draw_animated_vix_timeline, draw_3d_volatility,
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

pio.templates.default = "plotly_dark"

LANG_FILENAMES = {"th": "index.html", "en": "index.en.html"}


# ==========================================================
# LOCALIZED STRINGS — single source of truth for both languages
# ==========================================================
S = {
    "th": {
        # Header
        "html_lang": "th",
        "title": "🚨 Black Swan Risk Indicator — Live Report",
        "hero_h1": "🚨 Black Swan Risk Indicator",
        "hero_sub": "ระบบเตือนภัยล่วงหน้าก่อนวิกฤตการเงิน · รวม NLP sentiment + ML forecasting + regime detection",
        "updated_label": "📅 อัพเดทล่าสุด:",
        "data_window": "📊 ข้อมูลย้อนหลัง {years} ปี",
        "code_link": "ดูซอร์สโค้ดบน GitHub →",
        "lang_switch_label": "🌐 Languages:",
        "lang_other_label": "English",
        "lang_other_href": "index.en.html",

        # Section 1
        "sec1_h2": "🌤️ สรุปสภาพตลาดวันนี้",
        "sec1_tag": "ภาพรวม 30 วินาที — ตัวเลขที่นักลงทุนต้องจับตา + ความหมายเชิง regime",

        # Hero status messages
        "hero_safe_title": "Risk ระดับต่ำ — ตลาดอยู่ในโหมดปกติ",
        "hero_safe_sub": "VIX ต่ำ + Regime = Trending Bull → กลยุทธ์ลงทุนปกติยังใช้ได้",
        "hero_caution_title": "Risk ระดับกลาง — ต้องเฝ้าระวัง",
        "hero_caution_sub": "VIX เริ่มขยับ หรือ Regime ออกข้าง → ติดตามใกล้ชิด",
        "hero_danger_title": "Risk ระดับสูง — สัญญาณวิกฤต",
        "hero_danger_sub": "VIX > 30 หรือ Regime = Panic → พิจารณาลดความเสี่ยง",

        # Top facts
        "fact_vix_label": "VIX ปัจจุบัน",
        "fact_regime_label": "Market Regime",
        "fact_sp500_label": "S&P 500",
        "fact_sp500_desc": "ดัชนีหุ้นใหญ่อันดับ 500 ของอเมริกา",

        # Section 2 — VIX
        "sec2_h2": "🌡️ VIX = ดัชนีความกลัวของตลาด",
        "sec2_tag": "CBOE Volatility Index — คำนวณจาก implied vol ของ option S&P 500 30 วันข้างหน้า",
        "vix_explainer": """
        <div class="simple-list">
            <ul>
                <li><strong>VIX < 15</strong> = ตลาดสงบมาก (อาจ complacent เกินไป)</li>
                <li><strong>VIX 15-20</strong> = ระดับปกติของตลาด bull</li>
                <li><strong>VIX 20-30</strong> = นักลงทุนเริ่มกังวล</li>
                <li><strong>VIX 30-40</strong> = สัญญาณวิกฤต (ปี 2018 ขึ้นดอกเบี้ย, 2022 เงินเฟ้อ)</li>
                <li><strong>VIX > 40</strong> = วิกฤตจริง — สูงสุดในประวัติศาสตร์: <strong>82.69</strong> (16 มี.ค. 2020, COVID)</li>
            </ul>
        </div>
        """,
        "sec2_chart_title": "📈 VIX history + AI forecast (7 วันข้างหน้า)",
        "sec2_chart_caption": "💡 <strong>วิธีอ่าน</strong>: เส้นน้ำเงิน = VIX จริง · เส้นส้มประ = ค่าเฉลี่ย 20 วัน · ดาวแดง = จุดที่ XGBoost ทำนายว่า VIX จะอยู่ที่เท่าไหร่ในอีก 7 วัน",

        # Section 3 — Forecast
        "sec3_h2": "🔮 ML Forecast: ทำนาย VIX 7 วันข้างหน้า",
        "sec3_tag": "ใช้ XGBoost เทรนบน lag features ของ VIX + macro indicators (yield curve, gold, oil, DXY) — แต่ดูผลความแม่นใน section ถัดไปก่อน trust",
        "forecast_up_label": "พยากรณ์: VIX จะเพิ่มขึ้น",
        "forecast_up_msg": "โมเดลคาดว่า VIX จะอยู่ที่ {pred:.1f} (เพิ่ม {delta:+.1f} จากวันนี้) — เพิ่ม risk monitoring",
        "forecast_flat_label": "พยากรณ์: VIX ทรงตัว",
        "forecast_flat_msg": "VIX คาดว่าจะอยู่ที่ {pred:.1f} ใกล้เคียงปัจจุบัน — สถานการณ์ stable",
        "forecast_down_label": "พยากรณ์: VIX จะลดลง",
        "forecast_down_msg": "VIX คาดว่าจะลดเหลือ {pred:.1f} (ลด {delta:.1f}) — สัญญาณคลี่คลาย",
        "forecast_speech": "<strong>หมายเหตุสำคัญ</strong>: VIX เป็นตัวแปรที่พยากรณ์ยากมาก — แม้แต่ academic paper ก็มัก R² ใกล้ 0 ใช้ค่านี้เป็น <em>directional signal</em> ไม่ใช่ค่าเป๊ะ ดูผลความแม่นใน section ถัดไป",

        # Section 4 — Accuracy
        "sec4_h2": "🎯 แต่... โมเดลแม่นแค่ไหน?",
        "sec4_tag": "เราใช้ <strong>walk-forward cross-validation</strong> (TimeSeriesSplit) — มาตรฐาน time-series ที่ป้องกัน look-ahead bias",
        "accuracy_explainer": """
        <div class="simple-list">
            <ul>
                <li><strong>R²</strong> (R-squared) = วัดว่าโมเดลอธิบายข้อมูลได้ดีกว่าเดาค่าเฉลี่ยแค่ไหน</li>
                <li><strong>R² = 1.0</strong> = ทำนายเป๊ะทุกครั้ง (ไม่เคยเกิดในงานจริง)</li>
                <li><strong>R² = 0</strong> = เท่ากับเดาด้วยค่าเฉลี่ย — ไม่มีประโยชน์</li>
                <li><strong>R² ติดลบ</strong> = แย่กว่าเดาด้วยค่าเฉลี่ย — สัญญาณว่าโมเดลใช้ไม่ได้</li>
            </ul>
        </div>
        """,
        "acc_good_msg": "โมเดลทำได้ดีพอควร — มี predictive power เกินกว่า random",
        "acc_ok_msg": "โมเดลแม่นนิดหน่อย — ดีกว่าเดามั่วเล็กน้อย",
        "acc_bad_msg": "โมเดล <strong>ไม่สามารถทำนาย VIX ได้ดีกว่าเดาค่าเฉลี่ย</strong> — สอดคล้องกับ Random Walk Hypothesis ที่ Fama (Nobel 2013) เสนอ ตลาดมี randomness สูงมากจน ML ทำนายแย่กว่า baseline ง่ายๆ",

        # Section 5 — Model Comparison
        "sec5_h2": "⚔️ เปรียบเทียบโมเดล (Model Comparison)",
        "sec5_tag": "ทดสอบ 4 โมเดล ML + 1 Naive baseline ด้วย walk-forward CV — ใครชนะใน task พยากรณ์ VIX?",
        "models_list": """
        <div class="simple-list">
            <ul>
                <li><strong>👶 Naive baseline</strong>: persistence model — ทำนายว่า "VIX อีก 7 วัน = VIX วันนี้" (ไม่ใช้ ML เลย)</li>
                <li><strong>🌲 XGBoost</strong>: gradient-boosted trees, regularized (max_depth=3, n_estimators=100)</li>
                <li><strong>💡 LightGBM</strong>: tree-based แบบเร็ว มี leaf-wise growth</li>
                <li><strong>📐 Ridge</strong>: linear regression + L2 regularization</li>
                <li><strong>🧠 LSTM</strong>: 1-layer PyTorch LSTM (32 hidden units, dropout 0.2)</li>
            </ul>
        </div>
        """,
        "winner_label": "ผู้ชนะ",
        "winner_desc": "ผลแม่นที่สุดในการทดสอบครั้งนี้",
        "loser_label": "อันดับท้าย",
        "loser_desc": "ผลแย่สุดในรอบนี้",
        "cmp_naive_wins": "<strong>เซอร์ไพรส์! Naive baseline ชนะทุกโมเดล ML</strong><br><br>โมเดล \"VIX อีก 7 วัน = VIX วันนี้\" (R² = {naive_score:.3f}) ชนะ XGBoost, LightGBM, Ridge, และ LSTM ทุกตัว นี่คือ <strong>Random Walk Hypothesis</strong> ในการเงิน — ราคาสินทรัพย์เคลื่อนไหวแบบ stochastic สูง การทำนายระยะสั้นแทบเป็นไปไม่ได้<br><br><strong>บทเรียน Engineering</strong>: ทดสอบ naive baseline ก่อนเสมอ ถ้าโมเดล ML ไม่ชนะ persistence model = อย่า deploy",
        "cmp_ml_wins": "<strong>{best_model}</strong> ชนะ task นี้ (R² = {best_score:.3f}) แต่เราใส่ <strong>Naive baseline</strong> เป็นมาตรฐานเทียบ ทุกโมเดล ML ที่ deploy ในงานจริงควรชนะ persistence model — มิฉะนั้นอย่าใช้",

        # Section 6 — SHAP
        "sec6_h2": "🔍 SHAP: AI ตัดสินใจจากอะไร?",
        "sec6_tag": "<strong>SHAP (SHapley Additive exPlanations)</strong> ใช้ game theory บอกว่าแต่ละ feature ผลักดันการทำนายไปทางไหน — interpretable AI standard",
        "shap_explainer": """
        <div class="simple-list">
            <ul>
                <li>🎯 SHAP value คำนวณจาก <strong>Shapley values</strong> (Nobel economics 1953)</li>
                <li>📊 บอก contribution ของแต่ละ feature ต่อ prediction</li>
                <li>✅ <strong>Fair & consistent</strong> มากกว่า feature_importance ของ XGBoost ปกติ</li>
                <li>🔍 ใช้ debug model behavior ได้</li>
            </ul>
        </div>
        """,
        "shap_top_msg": "โมเดลพึ่ง <strong>{top_feat}</strong> มากที่สุด — เป็น dominant signal ในการทำนาย",

        # COVID
        "covid_h2": "🦠 Case Study: COVID-19 Crash 2020",
        "covid_tag": "ตัวอย่างจริง — ระบบ rule-based ของเราเตือนทันเหตุการณ์วิกฤตที่ใหญ่ที่สุดในรอบ 10 ปีหรือไม่?",
        "covid_first_signal": "สัญญาณเตือนครั้งแรก",
        "covid_lead_before": "<strong>{days} วันก่อน</strong>",
        "covid_lead_same": "ในช่วงเดียวกับ",
        "covid_lead_after": "{days} วันก่อน",
        "covid_lead_suffix": "VIX แตะจุดสูงสุด",
        "covid_peak": "VIX แตะจุดสูงสุด",
        "covid_peak_desc": "เมื่อ {date}",
        "covid_drawdown": "S&P 500 ตกลง",
        "covid_drawdown_desc": "จาก peak ถึง trough ในช่วงนี้",
        "covid_signals": "จำนวนสัญญาณเตือน",
        "covid_signals_desc": "ตลอด ก.พ. - มิ.ย. 2020",
        "covid_msg": "<strong>นี่คือสิ่งที่ระบบควรทำได้</strong> — เตือนล่วงหน้าก่อนวิกฤตจริง<br>กรณี COVID-19 ระบบส่งสัญญาณวันที่ <strong>{first_sig}</strong> {lead}จุดที่ VIX แตะ peak<br><br>หากนักลงทุนเชื่อสัญญาณ → ออกจากตลาดเป็นเงินสด → หลีกเลี่ยง drawdown <strong>{drawdown}%</strong> ภายใน 1 เดือน<br><br><em>แม้ ML forecasting ไม่แม่น แต่ rule-based detector (VIX > 30 + vol spike) ทำงานได้จริงในวิกฤต</em>",

        # Animated + 3D
        "anim_h2": "🎬 Animated Timeline",
        "anim_tag": "กดปุ่ม <strong>▶️ Play</strong> ดู VIX เคลื่อนไหวตามเดือนตลอด {years} ปี — เห็นทุกวิกฤตที่ผ่านมา (COVID, สงครามรัสเซีย-ยูเครน, เงินเฟ้อ 2022)",
        "anim_speech": "ลองสังเกตว่า VIX <strong>พุ่งทุกครั้งที่มี black swan event</strong>: ก.พ. 2020 (COVID), ก.พ. 2022 (สงคราม), ก.ย. 2022 (Fed ขึ้นดอกเบี้ย) Pattern ซ้ำๆ ที่เราพยายามจับ",
        "threed_h2": "🌐 3D Visualization",
        "threed_tag": "Scatter 3 มิติ: <strong>เวลา × S&P daily return × VIX</strong> หมุนด้วย mouse drag · Scroll zoom",
        "threed_speech": "<strong>Cluster ที่สังเกตได้</strong>: จุดแดง (VIX สูง) มักอยู่ในวันที่ S&P ลงแรง > 3% — ยืนยัน <strong>negative correlation</strong> ระหว่าง VIX กับ S&P return (มักอยู่ที่ -0.7 ถึง -0.8)",

        # Backtest
        "bt_h2": "💰 Backtest: กลยุทธ์ทำกำไรได้หรือไม่?",
        "bt_tag": "Simulation ย้อนหลัง {years} ปี — เริ่มทุน 1 ล้านบาท เปรียบเทียบ \"กลยุทธ์ของเรา\" vs \"Buy & Hold\" (รวม transaction cost {tx} bps)",
        "bt_strategy_label": "กลยุทธ์ Black Swan",
        "bt_baseline_label": "Buy & Hold (เปรียบเทียบ)",
        "bt_initial": "เริ่มต้น 1,000,000 บาท",
        "bt_return": "ผลตอบแทน",
        "bt_mdd": "Max Drawdown",
        "bt_strategy_fees": "💰 รวมค่าธรรมเนียมเทรด {n} ครั้ง ({tx} bps/turn) — ใกล้เคียงสภาพจริง",
        "bt_equity_h3": "📈 Equity Curve เปรียบเทียบ",
        "bt_dd_h3": "📉 Drawdown ใครต่ำกว่า?",
        "bt_dd_tag": "ลึกน้อย = เจ็บน้อย กลยุทธ์ที่ดีควรมี drawdown ตื้นกว่า benchmark",
        "bt_signal_h3": "🚨 สัญญาณเตือนเกิดเมื่อไหร่?",
        "bt_signal_tag": "จุดสามเหลี่ยมแดง = วันที่ rule-based detector ส่งสัญญาณ",
        "bt_win_msg": "🎉 กลยุทธ์ <strong>ชนะ</strong> Buy & Hold! ได้กำไรมากกว่า <strong>{diff:,.0f} บาท</strong> + drawdown ตื้นกว่าในช่วงตลาดตก",
        "bt_lose_msg": "❌ กลยุทธ์ <strong>แพ้</strong> Buy & Hold ในช่วงนี้ — เพราะ {years} ปีที่ผ่านมาตลาดเป็น bull market ระบบป้องกันจึงเสียกำไรจากการอยู่นอกตลาดในช่วง up trend<br><br>กลยุทธ์นี้จะ shine ในช่วงวิกฤตจริง เช่น COVID-2020 หรือ Subprime 2008",
        "ttest_h3": "🧪 Paired t-test: กลยุทธ์ดีกว่า Baseline อย่างมีนัยสำคัญหรือไม่?",
        "ttest_tag": "ทดสอบสมมติฐาน H₀: mean(strategy − baseline) ≤ 0 vs H₁: mean(strategy − baseline) > 0 (one-sided) ที่ α = 0.05",
        "ttest_reject": "✅ <strong>Reject H₀</strong> — กลยุทธ์ดีกว่า baseline อย่างมีนัยสำคัญทางสถิติ (p < α)",
        "ttest_fail": "⚠️ <strong>Fail to reject H₀</strong> — ไม่มีหลักฐานเพียงพอว่ากลยุทธ์ดีกว่า baseline (p ≥ α) ตรงกับ Random Walk Hypothesis",

        # Multi-asset
        "multi_h2": "🌍 Multi-Asset Volatility",
        "multi_tag": "เทียบ realized volatility ข้าม asset class: US equity, crypto, ทอง, ตลาดเกิดใหม่ — ถ้าทุก asset วุ่นวายพร้อมกัน = systemic crisis",
        "multi_chart_title": "Volatility across asset classes",
        "multi_x": "วันที่",
        "multi_y": "Volatility (VIX or realized vol proxy)",
        "multi_speech": "Bitcoin มี realized vol สูงกว่า S&P 500 ประมาณ 3-5 เท่า ขณะที่ <strong>ทอง</strong> เป็น \"safe haven\" — มัก uncorrelated หรือ negatively correlated กับ equity ในช่วงวิกฤต<br><br>ถ้าเห็น <strong>ทุก asset spike พร้อมกัน</strong> = สัญญาณ systemic risk (เช่น COVID-2020 ที่ทุก asset ตกพร้อมกัน รวมถึงทอง)",

        # Final summary
        "final_h2": "📖 บทสรุป",
        "final_now": "🌡️ <strong>สภาพตลาดวันนี้</strong>",
        "final_forecast": "🔮 <strong>คาดการณ์ 7 วันข้างหน้า</strong>",
        "final_use": "💡 <strong>ระบบนี้ใช้ทำอะไรได้</strong>",
        "final_use_text": "Early warning system สำหรับ market regime change รวม 3 signals: news sentiment (FinBERT) + ML forecast + rule-based crisis detector ไม่ใช่ระบบทำนายอนาคต 100% แต่ช่วย <strong>ลด drawdown</strong> ในช่วงวิกฤตได้",
        "disclaimer": "<strong>⚠️ Disclaimer</strong><br>โปรเจกต์เพื่อ <strong>การศึกษาและวิจัย</strong> เท่านั้น <strong>ไม่ใช่คำแนะนำการลงทุน</strong> — ผลในอดีตไม่การันตีอนาคต ตลาดอาจ irrational ได้นานกว่าที่คุณ solvent อยู่",

        # Tech
        "tech_label": "สำหรับ engineer / data scientist",
        "tech_h3": "📊 ตัวเลขเชิงเทคนิค",
        "tech_methodology_h3": "📐 Methodology",
        "tech_source": "Full source code:",

        # Footer
        "footer": "🚨 สร้างโดย <a href=\"https://github.com/oomNoNe/black-swan-indicator\">black-swan-indicator</a> · auto-rebuild ทุกสัปดาห์ผ่าน GitHub Actions · <a href=\"https://github.com/oomNoNe/black-swan-indicator/blob/main/README.th.md\">README ภาษาไทย</a>",

        # Regime labels
        "regime_bull": "Trending Bull",
        "regime_bull_desc": "ตลาด uptrend — SMA-50 > SMA-200 และ vol ต่ำ",
        "regime_panic": "Panic",
        "regime_panic_desc": "ตลาด downtrend + vol สูง — สัญญาณวิกฤต",
        "regime_ranging": "Ranging",
        "regime_ranging_desc": "ตลาด sideways — รอ breakout",
        "regime_unknown": "Unknown",
        "regime_unknown_desc": "ข้อมูลไม่พอ",

        # Logs (printed to console)
        "log_1": "📊 [1/9] โหลด macro data (cache 1h)...",
        "log_2": "📊 [2/9] วิเคราะห์ market regime...",
        "log_3": "🤖 [3/9] โหลด/เทรน XGBoost (cache 24h)...",
        "log_4": "📈 [4/9] Walk-forward validation (cache 24h)...",
        "log_5": "⚔️ [5/9] Model comparison (cache 24h)...",
        "log_6": "🔍 [6/9] SHAP values (cache 24h)...",
        "log_7": "💰 [7/9] Backtest with transaction costs...",
        "log_8": "🌍 [8/9] Multi-asset (cache 1h each)...",
        "log_9": "📚 [9/9] COVID-2020 case study...",
        "log_render": "📝 กำลังเขียน HTML report ({lang})...",
        "log_saved": "✅ Report saved:",
    },

    "en": {
        # Header
        "html_lang": "en",
        "title": "🚨 Black Swan Risk Indicator — Live Report",
        "hero_h1": "🚨 Black Swan Risk Indicator",
        "hero_sub": "Early-warning system for financial market crises · NLP sentiment + ML forecasting + regime detection",
        "updated_label": "📅 Last updated:",
        "data_window": "📊 {years} years of data",
        "code_link": "View source on GitHub →",
        "lang_switch_label": "🌐 Languages:",
        "lang_other_label": "ภาษาไทย",
        "lang_other_href": "index.html",

        # Section 1
        "sec1_h2": "🌤️ Today's Market Snapshot",
        "sec1_tag": "30-second overview — key indicators investors watch + regime interpretation",

        # Hero status
        "hero_safe_title": "Low Risk — Markets in Normal Mode",
        "hero_safe_sub": "Low VIX + Trending Bull regime → standard investment strategies remain viable",
        "hero_caution_title": "Moderate Risk — Monitor Closely",
        "hero_caution_sub": "VIX elevated or regime ranging → keep close watch on indicators",
        "hero_danger_title": "High Risk — Crisis Signals Active",
        "hero_danger_sub": "VIX > 30 or Panic regime → consider reducing risk exposure",

        # Top facts
        "fact_vix_label": "Current VIX",
        "fact_regime_label": "Market Regime",
        "fact_sp500_label": "S&P 500",
        "fact_sp500_desc": "Top 500 US listed companies index",

        # Section 2 — VIX
        "sec2_h2": "🌡️ VIX — The Market Fear Index",
        "sec2_tag": "CBOE Volatility Index — derived from S&P 500 option implied volatility, 30-day horizon",
        "vix_explainer": """
        <div class="simple-list">
            <ul>
                <li><strong>VIX < 15</strong> = very calm (possibly complacent)</li>
                <li><strong>VIX 15-20</strong> = normal bull market range</li>
                <li><strong>VIX 20-30</strong> = investors getting nervous</li>
                <li><strong>VIX 30-40</strong> = crisis signal (2018 rate hikes, 2022 inflation)</li>
                <li><strong>VIX > 40</strong> = full crisis — all-time peak: <strong>82.69</strong> (Mar 16, 2020 COVID)</li>
            </ul>
        </div>
        """,
        "sec2_chart_title": "📈 VIX history + AI forecast (7-day ahead)",
        "sec2_chart_caption": "💡 <strong>How to read</strong>: Blue line = actual VIX · Dotted orange = 20-day moving average · Red star = XGBoost's prediction for VIX 7 trading days from now",

        # Section 3 — Forecast
        "sec3_h2": "🔮 ML Forecast: 7-Day VIX Prediction",
        "sec3_tag": "XGBoost trained on VIX lag features + macro indicators (yield curve, gold, oil, DXY) — but check accuracy in the next section before trusting",
        "forecast_up_label": "Forecast: VIX rising",
        "forecast_up_msg": "Model expects VIX at {pred:.1f} (delta {delta:+.1f} from today) — heightened risk monitoring advised",
        "forecast_flat_label": "Forecast: VIX flat",
        "forecast_flat_msg": "VIX expected near {pred:.1f}, close to current level — stable conditions",
        "forecast_down_label": "Forecast: VIX falling",
        "forecast_down_msg": "VIX expected to drop to {pred:.1f} (delta {delta:.1f}) — easing signal",
        "forecast_speech": "<strong>Important caveat</strong>: VIX is notoriously difficult to forecast — even academic papers regularly report R² near zero. Treat this as a <em>directional signal</em>, not a precise number. See accuracy section below for the honest assessment.",

        # Section 4 — Accuracy
        "sec4_h2": "🎯 But... How Accurate Is It?",
        "sec4_tag": "Evaluated with <strong>walk-forward cross-validation</strong> (TimeSeriesSplit) — the gold standard for time-series ML that prevents look-ahead bias",
        "accuracy_explainer": """
        <div class="simple-list">
            <ul>
                <li><strong>R²</strong> (R-squared) = how much better the model explains variance vs predicting the mean</li>
                <li><strong>R² = 1.0</strong> = perfect prediction (never happens in practice)</li>
                <li><strong>R² = 0</strong> = same as predicting the mean — useless</li>
                <li><strong>R² negative</strong> = worse than predicting the mean — model is harmful</li>
            </ul>
        </div>
        """,
        "acc_good_msg": "Model has meaningful predictive power — outperforms random",
        "acc_ok_msg": "Model slightly better than random — marginal value",
        "acc_bad_msg": "Model <strong>cannot beat the mean prediction</strong> — consistent with the <strong>Random Walk Hypothesis</strong> (Fama, Nobel 2013). Markets contain enough randomness that ML often underperforms simple baselines on volatility forecasting.",

        # Section 5 — Model Comparison
        "sec5_h2": "⚔️ Model Comparison",
        "sec5_tag": "Tested 4 ML models + 1 Naive baseline with walk-forward CV — who wins at VIX forecasting?",
        "models_list": """
        <div class="simple-list">
            <ul>
                <li><strong>👶 Naive baseline</strong>: persistence model — predicts \"VIX in 7d = VIX today\" (no ML)</li>
                <li><strong>🌲 XGBoost</strong>: gradient-boosted trees, regularized (max_depth=3, n_estimators=100)</li>
                <li><strong>💡 LightGBM</strong>: fast tree-based with leaf-wise growth</li>
                <li><strong>📐 Ridge</strong>: linear regression with L2 regularization</li>
                <li><strong>🧠 LSTM</strong>: 1-layer PyTorch LSTM (32 hidden units, dropout 0.2)</li>
            </ul>
        </div>
        """,
        "winner_label": "Winner",
        "winner_desc": "Highest R² in this evaluation",
        "loser_label": "Worst",
        "loser_desc": "Lowest R² in this round",
        "cmp_naive_wins": "<strong>Surprise! Naive baseline beats every ML model</strong><br><br>The \"VIX in 7d = VIX today\" model (R² = {naive_score:.3f}) beats XGBoost, LightGBM, Ridge, and LSTM. This is the <strong>Random Walk Hypothesis</strong> in finance — asset prices move stochastically, making short-term prediction extremely difficult.<br><br><strong>Engineering lesson</strong>: Always test naive baselines first. If your ML doesn't beat persistence, don't deploy it.",
        "cmp_ml_wins": "<strong>{best_model}</strong> wins this task (R² = {best_score:.3f}). We included the <strong>Naive baseline</strong> as a reference — any production ML model should beat persistence, otherwise it's not worth deploying.",

        # Section 6 — SHAP
        "sec6_h2": "🔍 SHAP: What Drives Model Decisions?",
        "sec6_tag": "<strong>SHAP (SHapley Additive exPlanations)</strong> uses game theory to attribute predictions to features — the gold standard for interpretable AI",
        "shap_explainer": """
        <div class="simple-list">
            <ul>
                <li>🎯 SHAP values derived from <strong>Shapley values</strong> (Nobel Economics 1953)</li>
                <li>📊 Shows each feature's marginal contribution to a prediction</li>
                <li>✅ <strong>Fair & consistent</strong> — superior to XGBoost's built-in feature_importance</li>
                <li>🔍 Useful for debugging model behavior</li>
            </ul>
        </div>
        """,
        "shap_top_msg": "Model relies most heavily on <strong>{top_feat}</strong> — this is the dominant signal driving predictions",

        # COVID
        "covid_h2": "🦠 Case Study: COVID-19 Crash 2020",
        "covid_tag": "Real-world example — did our rule-based system catch the biggest crisis in a decade?",
        "covid_first_signal": "First signal fired",
        "covid_lead_before": "<strong>{days} days before</strong>",
        "covid_lead_same": "around",
        "covid_lead_after": "{days} days before",
        "covid_lead_suffix": "VIX peaked",
        "covid_peak": "VIX peak",
        "covid_peak_desc": "on {date}",
        "covid_drawdown": "S&P 500 drawdown",
        "covid_drawdown_desc": "peak-to-trough during this window",
        "covid_signals": "Total signals fired",
        "covid_signals_desc": "throughout Feb-Jun 2020",
        "covid_msg": "<strong>This is what the system is supposed to do</strong> — warn before the crisis fully unfolds.<br>For COVID-19, the first signal fired on <strong>{first_sig}</strong> {lead} the VIX peak.<br><br>An investor who acted on the signal → exited to cash → avoided ~<strong>{drawdown}%</strong> drawdown in 1 month.<br><br><em>Even though ML forecasting performed poorly, the rule-based detector (VIX > 30 + vol spike) worked in practice during a real crisis.</em>",

        # Animated + 3D
        "anim_h2": "🎬 Animated Timeline",
        "anim_tag": "Click <strong>▶️ Play</strong> to watch VIX evolve month-by-month over {years} years — every crisis is visible (COVID, Russia-Ukraine war, 2022 inflation)",
        "anim_speech": "Notice how VIX <strong>spikes during every black swan event</strong>: Feb 2020 (COVID), Feb 2022 (Russia war), Sep 2022 (Fed rate hikes). This is the recurring pattern we try to detect.",
        "threed_h2": "🌐 3D Visualization",
        "threed_tag": "3D scatter: <strong>time × S&P daily return × VIX</strong> — drag with mouse to rotate · scroll to zoom",
        "threed_speech": "<strong>Observable cluster</strong>: red points (high VIX) tend to coincide with days when S&P fell > 3% — confirming the well-known <strong>negative correlation</strong> between VIX and S&P returns (typically -0.7 to -0.8).",

        # Backtest
        "bt_h2": "💰 Backtest: Is the Strategy Profitable?",
        "bt_tag": "Historical simulation over {years} years — start with $1M, compare \"Black Swan Strategy\" vs \"Buy & Hold\" (including {tx} bps transaction costs)",
        "bt_strategy_label": "Black Swan Strategy",
        "bt_baseline_label": "Buy & Hold (baseline)",
        "bt_initial": "Initial $1,000,000",
        "bt_return": "Total return",
        "bt_mdd": "Max Drawdown",
        "bt_strategy_fees": "💰 Includes {n} trades with {tx} bps cost per turnover — realistic conditions",
        "bt_equity_h3": "📈 Equity Curve Comparison",
        "bt_dd_h3": "📉 Which strategy has shallower drawdown?",
        "bt_dd_tag": "Less depth = less pain. A good strategy should have shallower drawdown than the benchmark.",
        "bt_signal_h3": "🚨 When did signals fire?",
        "bt_signal_tag": "Red triangle markers = days the rule-based detector flagged risk",
        "bt_win_msg": "🎉 Strategy <strong>beats</strong> Buy & Hold! Earned <strong>${diff:,.0f}</strong> more + shallower drawdown during sell-offs.",
        "bt_lose_msg": "❌ Strategy <strong>underperforms</strong> Buy & Hold in this window — because the past {years} years were a bull market, defensive systems sacrifice upside by sitting in cash during rallies.<br><br>The strategy's value emerges during actual crises like COVID-2020 or 2008.",
        "ttest_h3": "🧪 Paired t-test: Is the strategy significantly better than baseline?",
        "ttest_tag": "Hypothesis test — H₀: mean(strategy − baseline) ≤ 0 vs H₁: mean(strategy − baseline) > 0 (one-sided) at α = 0.05",
        "ttest_reject": "✅ <strong>Reject H₀</strong> — strategy is statistically significantly better than baseline (p < α)",
        "ttest_fail": "⚠️ <strong>Fail to reject H₀</strong> — insufficient evidence that strategy beats baseline (p ≥ α). Consistent with the Random Walk Hypothesis.",

        # Multi-asset
        "multi_h2": "🌍 Multi-Asset Volatility",
        "multi_tag": "Realized volatility across asset classes: US equity, crypto, gold, emerging markets — if all assets spike together = systemic crisis",
        "multi_chart_title": "Volatility across asset classes",
        "multi_x": "Date",
        "multi_y": "Volatility (VIX or realized vol proxy)",
        "multi_speech": "Bitcoin's realized volatility is typically 3-5× higher than S&P 500. <strong>Gold</strong> acts as a \"safe haven\" — often uncorrelated or negatively correlated with equities during crises.<br><br>If <strong>all assets spike together</strong> = systemic risk signal (e.g., COVID-2020, when even gold fell briefly).",

        # Final summary
        "final_h2": "📖 Bottom Line",
        "final_now": "🌡️ <strong>Today's market</strong>",
        "final_forecast": "🔮 <strong>7-day forecast</strong>",
        "final_use": "💡 <strong>What this system does</strong>",
        "final_use_text": "An early-warning system for market regime changes. Combines 3 signals: news sentiment (FinBERT) + ML forecast + rule-based crisis detector. It does NOT predict the future with certainty — but it helps <strong>reduce drawdowns</strong> during crises.",
        "disclaimer": "<strong>⚠️ Disclaimer</strong><br>This is an <strong>educational and research project</strong> — <strong>not financial advice</strong>. Past performance does not guarantee future results. Markets can stay irrational longer than you can stay solvent.",

        # Tech
        "tech_label": "For engineers / data scientists",
        "tech_h3": "📊 Technical Numbers",
        "tech_methodology_h3": "📐 Methodology",
        "tech_source": "Full source code:",

        # Footer
        "footer": "🚨 Built by <a href=\"https://github.com/oomNoNe/black-swan-indicator\">black-swan-indicator</a> · auto-rebuilt weekly via GitHub Actions · <a href=\"https://github.com/oomNoNe/black-swan-indicator/blob/main/README.md\">English README</a>",

        # Regime labels
        "regime_bull": "Trending Bull",
        "regime_bull_desc": "Uptrend — SMA-50 > SMA-200 and low volatility",
        "regime_panic": "Panic",
        "regime_panic_desc": "Downtrend + high volatility — crisis signal",
        "regime_ranging": "Ranging",
        "regime_ranging_desc": "Sideways — waiting for breakout",
        "regime_unknown": "Unknown",
        "regime_unknown_desc": "Insufficient data",

        # Logs
        "log_1": "📊 [1/9] Loading macro data (cache 1h)...",
        "log_2": "📊 [2/9] Analyzing market regime...",
        "log_3": "🤖 [3/9] Loading/training XGBoost (cache 24h)...",
        "log_4": "📈 [4/9] Walk-forward validation (cache 24h)...",
        "log_5": "⚔️ [5/9] Model comparison (cache 24h)...",
        "log_6": "🔍 [6/9] SHAP values (cache 24h)...",
        "log_7": "💰 [7/9] Backtest with transaction costs...",
        "log_8": "🌍 [8/9] Multi-asset (cache 1h each)...",
        "log_9": "📚 [9/9] COVID-2020 case study...",
        "log_render": "📝 Rendering HTML report ({lang})...",
        "log_saved": "✅ Report saved:",
    },
}


# ==========================================================
# HELPERS
# ==========================================================
def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs='cdn',
                       config={'displayModeBar': False, 'responsive': True})


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


def vix_level_emoji(vix):
    if vix < 15: return "😎", "#00cc96"
    if vix < 20: return "☀️", "#00cc96"
    if vix < 25: return "⛅", "#FFA15A"
    if vix < 30: return "🌤️", "#FFA15A"
    if vix < 40: return "⛈️", "#EF553B"
    return "🌪️", "#EF553B"


def regime_info(regime, t):
    info = {
        "Trending Bull": ("📈", t["regime_bull"], t["regime_bull_desc"], "#00cc96"),
        "Panic": ("🔥", t["regime_panic"], t["regime_panic_desc"], "#EF553B"),
        "Ranging": ("⚖️", t["regime_ranging"], t["regime_ranging_desc"], "#FFA15A"),
        "Unknown": ("❓", t["regime_unknown"], t["regime_unknown_desc"], "#888"),
    }
    return info.get(regime, ("❓", regime, "", "#888"))


# ==========================================================
# ANALYSIS (data + models — language-agnostic)
# ==========================================================
def _train_forecaster(macro):
    fc = VIXForecaster("XGBoost")
    fc.train_model(macro)
    return fc


def run_analysis(t):
    print(t["log_1"])
    macro = cache_dataframe(f"macro_{YEARS_LOOKBACK}y",
                            lambda: fetch_macro_data(years=YEARS_LOOKBACK),
                            ttl_hours=1.0)
    if macro is None or macro.empty:
        raise RuntimeError("Could not load macro data")

    print(t["log_2"])
    regime = classify_market_regime(macro)

    print(t["log_3"])
    forecaster = cache_model(f"xgb_forecaster_{YEARS_LOOKBACK}y",
                             lambda: _train_forecaster(macro), ttl_hours=24.0)
    predicted_vix = forecaster.predict_vix(macro)
    current_vix = float(macro['VIX'].iloc[-1])

    print(t["log_4"])
    wf_result = cache_pickle(f"wf_xgb_{YEARS_LOOKBACK}y_{WF_SPLITS}f",
                             lambda: walk_forward_validate(macro, "XGBoost",
                                                           task="regression",
                                                           n_splits=WF_SPLITS),
                             ttl_hours=24.0)

    print(t["log_5"])
    cmp_reg = cache_pickle(f"cmp_reg_{YEARS_LOOKBACK}y_{WF_SPLITS}f",
                           lambda: compare_models(macro, task="regression",
                                                  n_splits=WF_SPLITS),
                           ttl_hours=24.0)

    print(t["log_6"])
    clean_df, feat_cols, _ = build_features(macro, classification=False)
    shap_data = cache_pickle(f"shap_{YEARS_LOOKBACK}y",
                             lambda: compute_shap_values(
                                 forecaster.model, clean_df[feat_cols]),
                             ttl_hours=24.0)
    shap_values, feat_names, X_sample = shap_data

    print(t["log_7"])
    bt = macro.copy()
    bt['Vol_20'] = bt['Close'].pct_change().rolling(20).std() * np.sqrt(252)
    bt['Vol_Median_252'] = bt['Vol_20'].rolling(252, min_periods=60).median()
    bt['Risk_Signal'] = (
        (bt['VIX'] > VIX_THRESHOLD) &
        (bt['Vol_20'] > bt['Vol_Median_252'] * VOL_MULTIPLIER)
    ).astype(int)
    bt_metrics = run_advanced_backtest(bt, transaction_cost_bps=TRANSACTION_COST_BPS)

    print(t["log_8"])
    assets = {}
    for name in ["S&P 500", "Bitcoin", "Gold", "Emerging Markets (EEM)"]:
        safe_key = name.replace(" ", "_").replace("(", "").replace(")", "").replace("&", "")
        df = cache_dataframe(f"asset_{safe_key}_{YEARS_LOOKBACK}y",
                             lambda n=name: fetch_asset_data(n, YEARS_LOOKBACK),
                             ttl_hours=1.0)
        if df is not None and not df.empty:
            assets[name] = df

    print(t["log_9"])
    covid_data = build_covid_case_study(bt, t)

    return {
        'macro': macro, 'regime': regime, 'forecaster': forecaster,
        'current_vix': current_vix, 'predicted_vix': predicted_vix,
        'wf_result': wf_result, 'cmp_reg': cmp_reg,
        'shap': (shap_values, feat_names, X_sample),
        'bt': bt, 'bt_metrics': bt_metrics, 'assets': assets,
        'covid': covid_data,
    }


def build_covid_case_study(bt_data, t):
    start = pd.Timestamp("2020-02-01")
    end = pd.Timestamp("2020-06-30")
    covid = bt_data.loc[(bt_data.index >= start) & (bt_data.index <= end)].copy()
    if covid.empty or len(covid) < 30:
        return None

    peak_idx = covid['VIX'].idxmax()
    peak_vix = float(covid['VIX'].max())
    signals = covid[covid['Risk_Signal'] == 1]
    first_signal = signals.index[0] if not signals.empty else None

    sp_peak = covid['Close'].iloc[0]
    sp_trough = covid['Close'].min()
    drawdown = ((sp_trough - sp_peak) / sp_peak) * 100
    lead_days = (peak_idx - first_signal).days if first_signal else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=covid.index, y=covid['Close'], mode='lines', name='S&P 500',
        line=dict(color='#1f77b4', width=2), yaxis='y',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>S&P: %{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=covid.index, y=covid['VIX'], mode='lines',
        name='VIX' if t["html_lang"] == "en" else 'VIX (ความกลัว)',
        line=dict(color='#EF553B', width=2, dash='dot'), yaxis='y2',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>VIX: %{y:.1f}<extra></extra>'
    ))
    if not signals.empty:
        signal_label = '🚨 Signal' if t["html_lang"] == "en" else '🚨 ระบบเตือน'
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['Close'], mode='markers', name=signal_label,
            marker=dict(color='red', size=12, symbol='triangle-down',
                        line=dict(color='white', width=1)), yaxis='y',
            hovertemplate='<b>🚨 %{x|%d %b %Y}</b><br>S&P: %{y:,.0f}<extra></extra>'
        ))
    if first_signal:
        ann_text = (f"🚨 First signal<br>{first_signal.strftime('%d %b %Y')}"
                    if t["html_lang"] == "en"
                    else f"🚨 เตือนครั้งแรก<br>{first_signal.strftime('%d %b %Y')}")
        fig.add_annotation(
            x=first_signal, y=covid.loc[first_signal, 'Close'],
            text=ann_text, showarrow=True, arrowhead=2, arrowcolor="red",
            ax=-80, ay=-60, bgcolor="rgba(239,85,59,0.2)",
            bordercolor="red", borderwidth=1, font=dict(color="white"),
        )
    peak_ann = (f"⛈️ VIX peak {peak_vix:.0f}<br>{peak_idx.strftime('%d %b %Y')}"
                if t["html_lang"] == "en"
                else f"⛈️ VIX สูงสุด {peak_vix:.0f}<br>{peak_idx.strftime('%d %b %Y')}")
    fig.add_annotation(
        x=peak_idx, y=peak_vix, text=peak_ann,
        showarrow=True, arrowhead=2, arrowcolor="orange",
        ax=80, ay=-30, yref='y2', bgcolor="rgba(255,161,90,0.2)",
        bordercolor="orange", borderwidth=1, font=dict(color="white"),
    )

    title = ("🦠 COVID-19 Crash (Feb - Jun 2020)" if t["html_lang"] == "en"
             else "🦠 COVID-19 Crash (กุมภาพันธ์ - มิถุนายน 2020)")
    fig.update_layout(
        title=title, template="plotly_dark", hovermode="x unified", height=500,
        xaxis=dict(title=None),
        yaxis=dict(title="S&P 500", side="left"),
        yaxis2=dict(title="VIX", side="right", overlaying="y",
                    showgrid=False, color="#EF553B"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(t=60, b=40),
    )

    return {
        'fig': fig, 'first_signal_date': first_signal,
        'peak_vix_date': peak_idx, 'peak_vix': peak_vix,
        'drawdown_pct': float(drawdown),
        'signal_lead_days': lead_days,
        'n_signals': int(signals.shape[0]),
    }


# ==========================================================
# HTML TEMPLATE — bilingual via {placeholders}
# ==========================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Sukhumvit Set", "Noto Sans Thai", sans-serif;
    background: #0e1117; color: #fafafa; line-height: 1.75; font-size: 16px;
  }}
  .container {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px; }}
  header {{
    text-align: center; padding: 44px 24px; margin-bottom: 32px;
    background: linear-gradient(135deg, rgba(31,119,180,0.08), rgba(239,85,59,0.08));
    border-radius: 16px;
  }}
  h1 {{ font-size: 2.5rem; margin-bottom: 10px; line-height: 1.2; }}
  .hero-sub {{ color: #aaa; font-size: 1.1rem; max-width: 720px; margin: 0 auto; }}
  .meta-row {{
    display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;
    margin-top: 22px; font-size: 0.9rem; color: #888;
  }}
  .meta-row a {{ color: #1f77b4; text-decoration: none; }}
  .meta-row a:hover {{ text-decoration: underline; }}
  .lang-switch {{
    display: inline-block; margin-top: 10px; padding: 6px 14px;
    background: rgba(255,255,255,0.05); border-radius: 8px;
    font-size: 0.9rem; color: #aaa;
  }}
  .lang-switch a {{ color: #1f77b4; font-weight: 500; }}
  section {{ margin: 52px 0; }}
  h2 {{ font-size: 1.7rem; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; }}
  .section-tagline {{ color: #aaa; font-size: 1rem; margin-bottom: 20px; }}
  h3 {{ font-size: 1.2rem; margin: 24px 0 12px; color: #ddd; }}
  .big-status {{ padding: 36px 28px; text-align: center; border-radius: 14px; margin: 20px 0; }}
  .big-emoji {{ font-size: 4.5rem; line-height: 1; margin-bottom: 14px; }}
  .big-title {{ font-size: 1.7rem; font-weight: 700; margin-bottom: 8px; }}
  .big-subtitle {{ font-size: 1.05rem; color: #ddd; }}
  .facts-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 14px; margin: 22px 0;
  }}
  .fact-card {{
    background: rgba(255,255,255,0.04); padding: 18px; border-radius: 10px;
    display: flex; gap: 14px; align-items: flex-start;
  }}
  .fact-emoji {{ font-size: 2.2rem; line-height: 1; }}
  .fact-label {{ font-size: 0.85rem; color: #aaa; }}
  .fact-value {{ font-size: 1.6rem; font-weight: 700; margin: 4px 0; }}
  .fact-explain {{ font-size: 0.92rem; color: #bbb; }}
  .speech-bubble {{
    background: rgba(31,119,180,0.12); border-left: 4px solid #1f77b4;
    padding: 16px 20px; border-radius: 8px; margin: 18px 0;
    display: flex; gap: 14px; align-items: flex-start;
  }}
  .bubble-emoji {{ font-size: 1.6rem; line-height: 1; }}
  .bubble-text {{ flex: 1; font-size: 1rem; line-height: 1.7; }}
  .bubble-text strong {{ color: #1f77b4; }}
  .compare-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 22px 0;
  }}
  @media (max-width: 700px) {{ .compare-grid {{ grid-template-columns: 1fr; }} }}
  .compare-card {{ padding: 22px; border-radius: 12px; text-align: center; }}
  .compare-card.winner {{ background: rgba(0,204,150,0.12); border: 2px solid #00cc96; }}
  .compare-card.loser {{ background: rgba(239,85,59,0.08); border: 2px solid rgba(239,85,59,0.4); }}
  .compare-emoji {{ font-size: 2.6rem; }}
  .compare-name {{ font-size: 1.2rem; font-weight: 700; margin: 6px 0; }}
  .compare-num {{ font-size: 2.2rem; font-weight: 700; margin: 6px 0; }}
  .compare-desc {{ color: #ccc; font-size: 0.95rem; }}
  .simple-list {{ background: rgba(255,255,255,0.03); padding: 18px 26px; border-radius: 10px; margin: 18px 0; }}
  .simple-list li {{ margin: 10px 0; list-style: none; padding-left: 28px; position: relative; }}
  .simple-list li::before {{ content: "👉"; position: absolute; left: 0; top: 0; }}
  .chart-wrap {{ background: rgba(255,255,255,0.02); padding: 14px; border-radius: 12px; margin: 18px 0; }}
  .final-summary {{
    margin-top: 48px; padding: 30px;
    background: linear-gradient(135deg, rgba(31,119,180,0.13), rgba(0,204,150,0.07));
    border-radius: 14px;
  }}
  .summary-item {{ margin: 14px 0; font-size: 1.05rem; line-height: 1.75; }}
  footer {{
    margin-top: 56px; padding-top: 22px; border-top: 1px solid rgba(255,255,255,0.1);
    color: #888; font-size: 0.88rem; text-align: center;
  }}
  footer a {{ color: #1f77b4; text-decoration: none; }}
  .nerd-mode {{ margin-top: 28px; padding: 16px; background: rgba(255,255,255,0.03); border-radius: 8px; }}
  .nerd-mode summary {{ cursor: pointer; font-weight: 600; color: #aaa; padding: 6px 0; list-style: none; }}
  .nerd-mode summary::-webkit-details-marker {{ display: none; }}
  .nerd-mode summary::before {{ content: "🤓 "; }}
  .nerd-mode[open] summary::before {{ content: "👇 "; }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>{hero_h1}</h1>
  <p class="hero-sub">{hero_sub}</p>
  <div class="meta-row">
    <span>{updated_label} <strong>{timestamp}</strong></span>
    <span>{data_window_resolved}</span>
    <span><a href="https://github.com/oomNoNe/black-swan-indicator" target="_blank">{code_link}</a></span>
  </div>
  <div class="lang-switch">
    {lang_switch_label} <strong>{current_lang_name}</strong> · <a href="{lang_other_href}">{lang_other_label}</a>
  </div>
</header>

<section>
  <h2>{sec1_h2}</h2>
  <p class="section-tagline">{sec1_tag}</p>
  {hero_status}
  <div class="facts-grid">{top_facts}</div>
</section>

<section>
  <h2>{sec2_h2}</h2>
  <p class="section-tagline">{sec2_tag}</p>
  {vix_explainer}
  {speech_vix}
  <h3>{sec2_chart_title}</h3>
  <div class="chart-wrap">{vix_chart}</div>
  <p class="section-tagline">{sec2_chart_caption}</p>
</section>

<section>
  <h2>{sec3_h2}</h2>
  <p class="section-tagline">{sec3_tag}</p>
  {forecast_card}
  {speech_forecast}
</section>

<section>
  <h2>{sec4_h2}</h2>
  <p class="section-tagline">{sec4_tag}</p>
  {accuracy_explainer}
  <div class="chart-wrap">{wf_chart}</div>
  {speech_accuracy}
</section>

<section>
  <h2>{sec5_h2}</h2>
  <p class="section-tagline">{sec5_tag}</p>
  {models_list}
  <div class="chart-wrap">{cmp_chart}</div>
  {compare_winner_loser}
  {speech_compare}
</section>

<section>
  <h2>{sec6_h2}</h2>
  <p class="section-tagline">{sec6_tag}</p>
  {shap_explainer}
  <div class="chart-wrap">{shap_chart}</div>
  {speech_shap}
</section>

{covid_section}

<section>
  <h2>{anim_h2}</h2>
  <p class="section-tagline">{anim_tag_resolved}</p>
  <div class="chart-wrap">{animated_chart}</div>
  {speech_animated}
</section>

<section>
  <h2>{threed_h2}</h2>
  <p class="section-tagline">{threed_tag}</p>
  <div class="chart-wrap">{volatility_3d_chart}</div>
  {speech_threed}
</section>

<section>
  <h2>{bt_h2}</h2>
  <p class="section-tagline">{bt_tag_resolved}</p>
  {backtest_cards}
  <h3>{bt_equity_h3}</h3>
  <div class="chart-wrap">{equity_chart}</div>
  <h3>{bt_dd_h3}</h3>
  <p class="section-tagline">{bt_dd_tag}</p>
  <div class="chart-wrap">{dd_chart}</div>
  <h3>{bt_signal_h3}</h3>
  <p class="section-tagline">{bt_signal_tag}</p>
  <div class="chart-wrap">{signal_chart}</div>
  {speech_backtest}

  <h3>{ttest_h3}</h3>
  <p class="section-tagline">{ttest_tag}</p>
  {ttest_card}
</section>

<section>
  <h2>{multi_h2}</h2>
  <p class="section-tagline">{multi_tag}</p>
  <div class="chart-wrap">{multi_asset_chart}</div>
  {speech_multi_asset}
</section>

<div class="final-summary">
  <h2>{final_h2}</h2>
  <div class="summary-item">{final_now}<br>{final_status}</div>
  <div class="summary-item">{final_forecast}<br>{final_forecast_text}</div>
  <div class="summary-item">{final_use}<br>{final_use_text}</div>
  <div class="summary-item" style="color: #FFA15A;">{disclaimer}</div>
</div>

<details class="nerd-mode">
  <summary>{tech_label}</summary>
  <div style="padding: 14px 0; line-height: 1.8;">
    <h3>{tech_h3}</h3>
    <ul style="padding-left: 24px;">
      <li><strong>VIX</strong>: {tech_vix}</li>
      <li><strong>AI Forecast (7d, XGBoost)</strong>: {tech_pred}</li>
      <li><strong>Walk-Forward Mean R²</strong>: {tech_r2} ± {tech_r2_std} ({wf_splits} folds)</li>
      <li><strong>Best Model (regression)</strong>: {tech_best_model}</li>
      <li><strong>Backtest Sharpe</strong>: Strategy {tech_strat_sharpe} vs B&H {tech_base_sharpe}</li>
      <li><strong>Max Drawdown</strong>: Strategy {tech_strat_mdd}% vs B&H {tech_base_mdd}%</li>
      <li><strong>Transaction cost</strong>: {tx_cost} bps per turnover</li>
      <li><strong>Features</strong>: 13 (VIX lags, S&P returns, yield curve, gold/oil/DXY)</li>
    </ul>
    <h3>{tech_methodology_h3}</h3>
    <ul style="padding-left: 24px;">
      <li>Time-series validation: TimeSeriesSplit, no shuffle (no look-ahead bias)</li>
      <li>FinBERT (ProsusAI) for news sentiment (separate Streamlit app)</li>
      <li>SHAP TreeExplainer for feature attribution</li>
      <li>Multi-asset volatility: realized vol 20d annualized (proxy for non-VIX assets)</li>
      <li>Regime: SMA-50/200 crossover + 20d rolling vol vs historical median</li>
    </ul>
    <p style="margin-top: 14px;">{tech_source}
      <a href="https://github.com/oomNoNe/black-swan-indicator" target="_blank">github.com/oomNoNe/black-swan-indicator</a>
    </p>
  </div>
</details>

<footer><p>{footer}</p></footer>

</div>
</body>
</html>
"""


# ==========================================================
# BUILD (per language)
# ==========================================================
def build(lang="th"):
    t = S[lang]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / LANG_FILENAMES[lang]

    data = run_analysis(t)
    print(t["log_render"].format(lang=lang))

    # ---- Hero status ----
    vix = data['current_vix']
    if vix < 20 and data['regime'] == "Trending Bull":
        hero_status = big_status("☀️", t["hero_safe_title"], t["hero_safe_sub"], "#00cc96")
    elif vix > 30 or data['regime'] == "Panic":
        hero_status = big_status("⛈️", t["hero_danger_title"], t["hero_danger_sub"], "#EF553B")
    else:
        hero_status = big_status("⛅", t["hero_caution_title"], t["hero_caution_sub"], "#FFA15A")

    # ---- Top facts ----
    vix_emoji, vix_color = vix_level_emoji(vix)
    regime_emoji, regime_name, regime_desc, regime_color = regime_info(data['regime'], t)
    sp_pct_20d = (data['macro']['Close'].iloc[-1] / data['macro']['Close'].iloc[-21] - 1) * 100

    top_facts = (
        fact_card("🌡️", t["fact_vix_label"], f"{vix:.2f}", vix_emoji)
        + fact_card("🎭", t["fact_regime_label"], f"{regime_emoji} {regime_name}", regime_desc)
        + fact_card("📊", t["fact_sp500_label"],
                    f"{data['macro']['Close'].iloc[-1]:,.0f}",
                    f"{sp_pct_20d:+.2f}% (20 day)")
    )

    # ---- VIX speech ----
    speech_vix = speech_bubble("💬", t["sec2_chart_caption"])

    # ---- Forecast card ----
    pred = data['predicted_vix']
    delta = pred - vix
    if delta > 2:
        fc_label, fc_msg, fc_color, fc_emoji = (t["forecast_up_label"],
            t["forecast_up_msg"].format(pred=pred, delta=delta), "#EF553B", "📈")
    elif delta > -0.5:
        fc_label, fc_msg, fc_color, fc_emoji = (t["forecast_flat_label"],
            t["forecast_flat_msg"].format(pred=pred), "#FFA15A", "↔️")
    else:
        fc_label, fc_msg, fc_color, fc_emoji = (t["forecast_down_label"],
            t["forecast_down_msg"].format(pred=pred, delta=delta), "#00cc96", "📉")
    forecast_card = big_status(fc_emoji, fc_label, fc_msg, fc_color)
    speech_forecast = speech_bubble("💬", t["forecast_speech"])

    # ---- Accuracy ----
    r2 = data['wf_result']['mean_score']
    if r2 > 0.3:
        acc_msg = t["acc_good_msg"]
    elif r2 > 0:
        acc_msg = t["acc_ok_msg"]
    else:
        acc_msg = t["acc_bad_msg"]
    speech_accuracy = speech_bubble("💬", acc_msg)
    wf_chart = fig_to_html(draw_walkforward_chart(data['wf_result']['predictions_df']))

    # ---- Comparison ----
    cmp_fig = draw_model_comparison(data['cmp_reg'])
    cmp_chart = fig_to_html(cmp_fig) if cmp_fig else "<p>n/a</p>"
    cmp_sorted = data['cmp_reg'].sort_values('Mean Score', ascending=False).reset_index(drop=True)
    best = cmp_sorted.iloc[0]
    worst = cmp_sorted.iloc[-1]
    compare_winner_loser = f"""
    <div class="compare-grid">
        <div class="compare-card winner">
            <div class="compare-emoji">🥇</div>
            <div class="compare-name">{best['Model']}</div>
            <div class="compare-num">R² = {best['Mean Score']:.3f}</div>
            <div class="compare-desc">{t['winner_desc']}</div>
        </div>
        <div class="compare-card loser">
            <div class="compare-emoji">🥉</div>
            <div class="compare-name">{worst['Model']}</div>
            <div class="compare-num">R² = {worst['Mean Score']:.3f}</div>
            <div class="compare-desc">{t['loser_desc']}</div>
        </div>
    </div>
    """
    naive_row = data['cmp_reg'][data['cmp_reg']['Model'] == 'Naive']
    if not naive_row.empty and best['Model'] == 'Naive':
        speech_compare = speech_bubble("💬",
            t["cmp_naive_wins"].format(naive_score=float(naive_row.iloc[0]['Mean Score'])))
    else:
        speech_compare = speech_bubble("💬",
            t["cmp_ml_wins"].format(best_model=best['Model'], best_score=best['Mean Score']))

    # ---- SHAP ----
    shap_values, feat_names, X_sample = data['shap']
    if shap_values is not None:
        shap_chart = fig_to_html(draw_shap_summary(shap_values, feat_names, X_sample))
        importance = np.abs(shap_values).mean(axis=0)
        top_feat = feat_names[int(np.argmax(importance))] if feat_names else "?"
        speech_shap = speech_bubble("💬", t["shap_top_msg"].format(top_feat=top_feat))
    else:
        shap_chart = "<p>n/a</p>"
        speech_shap = ""

    # ---- COVID section ----
    covid = data.get('covid')
    if covid:
        if covid['signal_lead_days'] and covid['signal_lead_days'] > 0:
            lead_text = t["covid_lead_before"].format(days=covid['signal_lead_days'])
        else:
            lead_text = t["covid_lead_same"]
        first_sig_text = (covid['first_signal_date'].strftime('%d %b %Y')
                          if covid['first_signal_date'] else "n/a")
        peak_text = covid['peak_vix_date'].strftime('%d %b %Y')
        lead_with_suffix = f"{lead_text} {t['covid_lead_suffix']}"

        covid_section = f"""
<section>
  <h2>{t['covid_h2']}</h2>
  <p class="section-tagline">{t['covid_tag']}</p>
  <div class="facts-grid">
    {fact_card("🚨", t['covid_first_signal'], first_sig_text, lead_with_suffix)}
    {fact_card("⛈️", t['covid_peak'], f"{covid['peak_vix']:.0f}", t['covid_peak_desc'].format(date=peak_text))}
    {fact_card("📉", t['covid_drawdown'], f"{covid['drawdown_pct']:.1f}%", t['covid_drawdown_desc'])}
    {fact_card("🔔", t['covid_signals'], f"{covid['n_signals']}", t['covid_signals_desc'])}
  </div>
  <div class="chart-wrap">{fig_to_html(covid['fig'])}</div>
  {speech_bubble("💬", t['covid_msg'].format(first_sig=first_sig_text, lead=lead_text, drawdown=f"{abs(covid['drawdown_pct']):.0f}"))}
</section>
"""
    else:
        covid_section = ""

    # ---- Animated + 3D ----
    animated_chart = fig_to_html(draw_animated_vix_timeline(data['macro'], step_months=3))
    volatility_3d_chart = fig_to_html(draw_3d_volatility(data['macro']))
    speech_animated = speech_bubble("💬", t["anim_speech"])
    speech_threed = speech_bubble("💬", t["threed_speech"])

    # ---- Backtest ----
    strat = data['bt_metrics']["Black_Swan_Strategy"]
    base = data['bt_metrics']["Baseline_Buy_Hold"]
    ts = data['bt_metrics']["Trading_Stats"]
    bt_clean = data['bt'].dropna(subset=['Market_Return', 'Strategy_Return'])
    strat_pct = ((1 + bt_clean['Strategy_Return']).prod() - 1) * 100
    base_pct = ((1 + bt_clean['Market_Return']).prod() - 1) * 100
    strat_final = 1_000_000 * (1 + strat_pct / 100)
    base_final = 1_000_000 * (1 + base_pct / 100)

    backtest_cards = f"""
    <div class="compare-grid">
        <div class="compare-card {'winner' if strat_pct > base_pct else 'loser'}">
            <div class="compare-emoji">🛡️</div>
            <div class="compare-name">{t['bt_strategy_label']}</div>
            <div class="compare-num">{strat_final:,.0f}</div>
            <div class="compare-desc">{t['bt_initial']} → {strat_pct:+.1f}%<br>{t['bt_mdd']}: <strong>{strat['Max Drawdown (%)']:.1f}%</strong></div>
        </div>
        <div class="compare-card {'winner' if base_pct > strat_pct else 'loser'}">
            <div class="compare-emoji">📊</div>
            <div class="compare-name">{t['bt_baseline_label']}</div>
            <div class="compare-num">{base_final:,.0f}</div>
            <div class="compare-desc">{t['bt_initial']} → {base_pct:+.1f}%<br>{t['bt_mdd']}: <strong>{base['Max Drawdown (%)']:.1f}%</strong></div>
        </div>
    </div>
    <p class="section-tagline" style="text-align: center;">{t['bt_strategy_fees'].format(n=ts['Number of Trades'], tx=TRANSACTION_COST_BPS)}</p>
    """
    equity_chart = fig_to_html(draw_equity_curve_chart(data['bt']))
    dd_chart = fig_to_html(draw_drawdown_chart(data['bt']))
    signal_chart = fig_to_html(draw_backtest_chart(data['bt']))
    if strat_pct > base_pct:
        speech_backtest = speech_bubble("💬",
            t["bt_win_msg"].format(diff=strat_final - base_final))
    else:
        speech_backtest = speech_bubble("💬",
            t["bt_lose_msg"].format(years=YEARS_LOOKBACK))

    # ===== Paired t-test card =====
    ttest = data['bt_metrics'].get("Statistical_Test", {})
    if ttest and not pd.isna(ttest.get("t_statistic", np.nan)):
        verdict = t["ttest_reject"] if ttest["reject_h0"] else t["ttest_fail"]
        card_color = "#00cc96" if ttest["reject_h0"] else "#FFA15A"
        ttest_card = f"""
        <div style="background: linear-gradient(135deg, {card_color}22, {card_color}11); border: 2px solid {card_color}; padding: 20px 24px; border-radius: 12px; margin: 16px 0;">
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 14px;">
            <div><div style="color: #aaa; font-size: 0.85rem;">t-statistic</div><div style="font-size: 1.5rem; font-weight: 700;">{ttest['t_statistic']:.4f}</div></div>
            <div><div style="color: #aaa; font-size: 0.85rem;">p-value (one-sided)</div><div style="font-size: 1.5rem; font-weight: 700;">{ttest['p_value_one_sided']:.4f}</div></div>
            <div><div style="color: #aaa; font-size: 0.85rem;">α (significance)</div><div style="font-size: 1.5rem; font-weight: 700;">{ttest['alpha']}</div></div>
            <div><div style="color: #aaa; font-size: 0.85rem;">n (daily obs)</div><div style="font-size: 1.5rem; font-weight: 700;">{ttest['n_observations']:,}</div></div>
          </div>
          <div style="padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 1.05rem;">
            {verdict}
          </div>
        </div>
        """
    else:
        ttest_card = ""

    # ---- Multi-asset ----
    if data['assets']:
        fig = go.Figure()
        for name, df in data['assets'].items():
            fig.add_trace(go.Scatter(x=df.index, y=df['VIX'], mode='lines',
                                     name=name, line=dict(width=1.5)))
        fig.update_layout(
            title=t["multi_chart_title"],
            xaxis_title=t["multi_x"], yaxis_title=t["multi_y"],
            template="plotly_dark", height=420, hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        multi_asset_chart = fig_to_html(fig)
    else:
        multi_asset_chart = "<p>n/a</p>"
    speech_multi_asset = speech_bubble("💬", t["multi_speech"])

    # ---- Final summary ----
    final_status_text = f"{vix_emoji} <strong>{regime_name}</strong> · VIX = {vix:.2f}"
    if delta > 2:
        final_forecast_text = f"⚠️ VIX expected to rise to {pred:.1f} (+{delta:.1f})" if lang == "en" \
                              else f"⚠️ VIX คาดว่าจะขึ้นเป็น {pred:.1f} (+{delta:.1f})"
    elif delta > -0.5:
        final_forecast_text = f"↔️ VIX expected to stay near {pred:.1f}" if lang == "en" \
                              else f"↔️ VIX คาดว่าจะทรงตัวที่ {pred:.1f}"
    else:
        final_forecast_text = f"✅ VIX expected to drop to {pred:.1f} ({delta:.1f})" if lang == "en" \
                              else f"✅ VIX คาดว่าจะลดเป็น {pred:.1f} ({delta:.1f})"

    # ---- Render ----
    html = HTML_TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_window_resolved=t["data_window"].format(years=YEARS_LOOKBACK),
        anim_tag_resolved=t["anim_tag"].format(years=YEARS_LOOKBACK),
        bt_tag_resolved=t["bt_tag"].format(years=YEARS_LOOKBACK, tx=TRANSACTION_COST_BPS),
        current_lang_name=("English" if lang == "en" else "ไทย"),
        hero_status=hero_status, top_facts=top_facts,
        speech_vix=speech_vix,
        vix_chart=fig_to_html(draw_vix_history_chart(data['macro'], predicted_vix=data['predicted_vix'])),
        forecast_card=forecast_card, speech_forecast=speech_forecast,
        wf_chart=wf_chart, speech_accuracy=speech_accuracy,
        cmp_chart=cmp_chart, compare_winner_loser=compare_winner_loser,
        speech_compare=speech_compare, shap_chart=shap_chart, speech_shap=speech_shap,
        covid_section=covid_section,
        animated_chart=animated_chart, volatility_3d_chart=volatility_3d_chart,
        speech_animated=speech_animated, speech_threed=speech_threed,
        backtest_cards=backtest_cards, equity_chart=equity_chart,
        dd_chart=dd_chart, signal_chart=signal_chart, speech_backtest=speech_backtest,
        ttest_card=ttest_card,
        multi_asset_chart=multi_asset_chart, speech_multi_asset=speech_multi_asset,
        final_status=final_status_text, final_forecast_text=final_forecast_text,
        tech_vix=f"{vix:.2f}",
        tech_pred=f"{pred:.2f}",
        tech_r2=f"{r2:.3f}",
        tech_r2_std=f"{data['wf_result']['std_score']:.3f}",
        tech_best_model=best['Model'],
        tech_strat_sharpe=f"{strat['Sharpe Ratio']:.2f}",
        tech_base_sharpe=f"{base['Sharpe Ratio']:.2f}",
        tech_strat_mdd=f"{strat['Max Drawdown (%)']:.2f}",
        tech_base_mdd=f"{base['Max Drawdown (%)']:.2f}",
        wf_splits=WF_SPLITS, tx_cost=TRANSACTION_COST_BPS,
        **t,  # all other string keys from dict
    )

    output_file.write_text(html, encoding='utf-8')
    print(f"{t['log_saved']} {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Build static HTML report(s)")
    parser.add_argument("--lang", choices=["th", "en", "both"], default="both")
    args = parser.parse_args()

    langs = ["th", "en"] if args.lang == "both" else [args.lang]
    for lang in langs:
        print(f"\n{'=' * 50}\nBuilding {lang.upper()} version\n{'=' * 50}")
        build(lang)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
