import pandas as pd
import numpy as np
from scipy import stats


def calculate_professional_metrics(returns):
    """คำนวณตัวชี้วัด Quant Metrics ขั้นสูง"""
    if returns.empty or returns.std() == 0:
        return {"Sharpe Ratio": 0.0, "Max Drawdown (%)": 0.0, "Win Rate (%)": 0.0, "Profit Factor": 0.0}

    # Sharpe Ratio (สมมติ R_f = 0)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    # Max Drawdown
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


def run_advanced_backtest(df, transaction_cost_bps=0.0):
    """
    เปรียบเทียบกลยุทธ์ vs Buy & Hold

    Args:
        df: DataFrame with 'Close' and 'Risk_Signal' (0 or 1)
        transaction_cost_bps: ต้นทุนการเทรดต่อครั้ง (basis points)
                              0 = ไม่คิด (idealized)
                              5 = 0.05% (typical retail broker)
                              10 = 0.10% (conservative)
                              25 = 0.25% (high-cost broker / OTC)

    Returns dict with both strategies' metrics + turnover stats
    """
    try:
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided.")

        # Market return
        df['Market_Return'] = df['Close'].pct_change()

        # Position: 1 = ถือหุ้น, 0 = อยู่ในเงินสด
        # ใช้ shift(1) เพื่อหลีกเลี่ยง look-ahead bias
        df['Position'] = np.where(df['Risk_Signal'].shift(1) == 1, 0, 1)

        # ----------------------------------------------------------
        # Transaction cost calculation
        # ----------------------------------------------------------
        # คิด cost เฉพาะวันที่ position เปลี่ยน (entry/exit)
        # turnover = |position_t - position_{t-1}|  (0, 1, หรือ -1)
        df['Position_Change'] = df['Position'].diff().abs().fillna(0)

        # แปลง bps → decimal: 10 bps = 0.10% = 0.001
        cost_per_trade = transaction_cost_bps / 10_000.0

        # หักต้นทุนทุกวันที่มี turnover
        df['Cost'] = df['Position_Change'] * cost_per_trade

        # Strategy return = position × market return - cost
        df['Strategy_Return'] = (df['Position'] * df['Market_Return']) - df['Cost']

        df_clean = df.dropna(subset=['Market_Return', 'Strategy_Return'])

        strategy_metrics = calculate_professional_metrics(df_clean['Strategy_Return'])
        baseline_metrics = calculate_professional_metrics(df_clean['Market_Return'])

        # Turnover stats
        n_trades = int(df['Position_Change'].sum())
        total_cost_pct = float(df['Cost'].sum() * 100)
        days_in_market = int(df['Position'].sum())
        days_in_cash = len(df) - days_in_market

        # ----------------------------------------------------------
        # Paired t-test: Strategy vs Buy & Hold
        # ----------------------------------------------------------
        # H0: mean(strategy_return - market_return) = 0
        # H1: mean(strategy_return - market_return) > 0
        # alpha = 0.05 (one-sided)
        ttest_result = compare_strategies_ttest(
            df_clean['Strategy_Return'], df_clean['Market_Return']
        )

        return {
            "Black_Swan_Strategy": strategy_metrics,
            "Baseline_Buy_Hold": baseline_metrics,
            "Trading_Stats": {
                "Number of Trades": n_trades,
                "Total Cost (% of capital)": round(total_cost_pct, 3),
                "Days in Market": days_in_market,
                "Days in Cash": days_in_cash,
                "Transaction Cost (bps)": transaction_cost_bps,
            },
            "Statistical_Test": ttest_result,
        }
    except Exception as e:
        print(f"[Backtester Error] Failed to calculate metrics: {e}")
        return None


def compare_strategies_ttest(strategy_returns, baseline_returns, alpha=0.05):
    """
    Paired t-test เปรียบเทียบ strategy vs baseline บน daily returns

    H₀: mean(strategy - baseline) ≤ 0  (strategy ไม่ดีกว่า baseline)
    H₁: mean(strategy - baseline) > 0  (strategy ดีกว่า baseline)

    ใช้ paired (related samples) เพราะ daily returns ของทั้ง 2 มา
    จากตลาดเดียวกัน วันเดียวกัน → ไม่ independent

    Args:
        strategy_returns: pd.Series ของ daily returns กลยุทธ์
        baseline_returns: pd.Series ของ daily returns benchmark
        alpha: significance level (default 0.05)

    Returns:
        dict with t_statistic, p_value, conclusion, mean_diff, n_observations
    """
    s = pd.Series(strategy_returns).dropna()
    b = pd.Series(baseline_returns).dropna()
    aligned = pd.concat([s, b], axis=1, join='inner').dropna()
    aligned.columns = ['strategy', 'baseline']

    if len(aligned) < 30:
        return {
            "t_statistic": np.nan,
            "p_value_one_sided": np.nan,
            "alpha": alpha,
            "reject_h0": False,
            "conclusion": "Insufficient sample size (n < 30)",
            "mean_diff_daily_pct": np.nan,
            "n_observations": len(aligned),
        }

    diffs = aligned['strategy'] - aligned['baseline']

    # scipy.stats.ttest_rel: two-sided by default. We do one-sided manually.
    t_stat, p_two_sided = stats.ttest_rel(aligned['strategy'], aligned['baseline'])
    # One-sided p-value (alternative: strategy > baseline)
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - (p_two_sided / 2)

    reject_h0 = bool(p_one_sided < alpha)
    if reject_h0:
        conclusion = (f"Reject H₀ at α={alpha}: strategy returns are "
                      f"statistically significantly higher than baseline.")
    else:
        conclusion = (f"Fail to reject H₀ at α={alpha}: no statistically "
                      f"significant evidence that strategy beats baseline.")

    return {
        "t_statistic": float(round(t_stat, 4)),
        "p_value_one_sided": float(round(p_one_sided, 4)),
        "alpha": alpha,
        "reject_h0": reject_h0,
        "conclusion": conclusion,
        "mean_diff_daily_pct": float(round(diffs.mean() * 100, 4)),
        "n_observations": int(len(aligned)),
    }
