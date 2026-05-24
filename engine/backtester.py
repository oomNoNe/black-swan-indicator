import pandas as pd
import numpy as np


def calculate_professional_metrics(returns):
    """คำนวณตัวชี้วัด Quant Metrics ขั้นสูง"""
    if returns.empty or returns.std() == 0:
        return {"Sharpe Ratio": 0.0, "Max Drawdown (%)": 0.0, "Win Rate (%)": 0.0, "Profit Factor": 0.0}

    # Sharpe Ratio (สมมติให้อัตราผลตอบแทนปราศจากความเสี่ยง R_f = 0 เพื่อความเรียบง่าย)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    # Maximum Drawdown (MDD)
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min() * 100

    # Win Rate
    wins = len(returns[returns > 0])
    total_trades = len(returns[returns != 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0

    # Profit Factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

    return {
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2)
    }


def run_advanced_backtest(df):
    """เปรียบเทียบสัญญาณการเทรดกับเส้นฐาน (Buy & Hold)"""
    try:
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided.")

        # สมมติฐาน: 'Close' คือราคาของ S&P 500, 'Risk_Signal' คือ (1 = เตือนวิกฤต, 0 = ปกติ)
        df['Market_Return'] = df['Close'].pct_change()

        # กลยุทธ์: หากระบบเตือน (1) ให้ดึงเงินสดออก (ผลตอบแทน=0) หรือ Short-sell, หากปกติ (0) ให้ถือครองต่อไป
        df['Strategy_Return'] = df['Market_Return'] * np.where(df['Risk_Signal'].shift(1) == 1, 0, 1)

        df = df.dropna()

        strategy_metrics = calculate_professional_metrics(df['Strategy_Return'])
        baseline_metrics = calculate_professional_metrics(df['Market_Return'])

        return {
            "Black_Swan_Strategy": strategy_metrics,
            "Baseline_Buy_Hold": baseline_metrics
        }
    except Exception as e:
        print(f"[Backtester Error] Failed to calculate metrics: {e}")
        return None