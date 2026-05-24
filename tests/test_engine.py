"""Unit tests สำหรับ engine modules — ทดสอบ pure functions ที่ไม่พึ่ง network/model"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.backtester import calculate_professional_metrics, run_advanced_backtest
from engine.regime_detector import classify_market_regime, dynamic_risk_equation
from engine.ml_predictor import VIXForecaster


# ---------- Fixtures ----------
@pytest.fixture
def synthetic_market_data():
    """สร้างข้อมูล S&P 500 + VIX จำลอง 500 วัน"""
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    returns = np.random.normal(0.0005, 0.01, n)
    close = 4000 * np.exp(np.cumsum(returns))
    vix = np.clip(15 + np.random.normal(0, 5, n), 10, 60)
    return pd.DataFrame({"Close": close, "VIX": vix}, index=idx)


# ---------- backtester ----------
def test_metrics_empty_returns():
    result = calculate_professional_metrics(pd.Series(dtype=float))
    assert result["Sharpe Ratio"] == 0.0
    assert result["Max Drawdown (%)"] == 0.0


def test_metrics_positive_returns():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008])
    result = calculate_professional_metrics(returns)
    assert result["Sharpe Ratio"] > 0
    assert result["Win Rate (%)"] == 80.0


def test_backtest_runs_with_signal(synthetic_market_data):
    df = synthetic_market_data.copy()
    df["Risk_Signal"] = (df["VIX"] > 25).astype(int)
    result = run_advanced_backtest(df)
    assert result is not None
    assert "Black_Swan_Strategy" in result
    assert "Baseline_Buy_Hold" in result


# ---------- regime detector ----------
def test_regime_returns_unknown_on_short_data():
    df = pd.DataFrame({"Close": [100, 101, 102]})
    assert classify_market_regime(df) == "Unknown"


def test_regime_classifies_known_regime(synthetic_market_data):
    regime = classify_market_regime(synthetic_market_data)
    assert regime in {"Trending Bull", "Ranging", "Panic", "Unknown"}


def test_regime_no_side_effect(synthetic_market_data):
    """regime_detector ต้องไม่แก้ DataFrame ต้นทาง (bug ที่เคยแก้)"""
    cols_before = set(synthetic_market_data.columns)
    _ = classify_market_regime(synthetic_market_data)
    assert set(synthetic_market_data.columns) == cols_before


def test_dynamic_risk_equation_weights():
    # Panic ควรให้น้ำหนักข่าวมาก (0.7) → ผลใกล้ news_risk
    score_panic = dynamic_risk_equation(news_risk=100, market_risk=0, regime="Panic")
    score_bull = dynamic_risk_equation(news_risk=100, market_risk=0, regime="Trending Bull")
    assert score_panic > score_bull
    assert score_panic == 70.0
    assert score_bull == 30.0


# ---------- ml_predictor ----------
def test_predictor_uninitialized_returns_string():
    forecaster = VIXForecaster()
    result = forecaster.predict_vix(pd.DataFrame())
    assert result == "Uninitialized"


def test_predictor_train_and_predict(synthetic_market_data):
    forecaster = VIXForecaster()
    status = forecaster.train_model(synthetic_market_data)
    assert status["status"] == "success"
    assert forecaster.is_trained is True

    prediction = forecaster.predict_vix(synthetic_market_data)
    assert isinstance(prediction, (float, np.floating))
    assert 0 < float(prediction) < 200  # sanity range
