import pandas as pd
import numpy as np


def classify_market_regime(df):
    """วิเคราะห์และตรวจจับสภาพตลาดปัจจุบัน"""
    try:
        if df.empty or len(df) < 200:
            return "Unknown"

        # ทำงานบนสำเนาเพื่อไม่ให้กระทบ DataFrame ต้นทาง
        df = df.copy()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['Vol_20'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

        curr_sma50 = df['SMA_50'].iloc[-1]
        curr_sma200 = df['SMA_200'].iloc[-1]
        curr_vol = df['Vol_20'].iloc[-1]

        # ใช้มัธยฐานความผันผวนของตลาดตลอดกาลเป็นเกณฑ์แบ่งแยก (Dynamic Threshold)
        vol_threshold = df['Vol_20'].median()

        # นำหลักการตรวจจับของ Regime มาประยุกต์ใช้
        if curr_sma50 > curr_sma200 and curr_vol < vol_threshold:
            return "Trending Bull"
        elif curr_sma50 < curr_sma200 and curr_vol > vol_threshold:
            return "Panic"
        else:
            return "Ranging"

    except Exception as e:
        print(f"[Regime Detection Error] Failed to classify regime: {e}")
        return "Unknown"


def dynamic_risk_equation(news_risk, market_risk, regime):
    """ปรับน้ำหนักค่าของ The Crisis Equation แบบไดนามิกตามสภาพตลาด"""
    # กำหนดน้ำหนักเริ่มต้น (w_news, w_market)
    weights = {"Trending Bull": (0.3, 0.7), "Panic": (0.7, 0.3), "Ranging": (0.5, 0.5), "Unknown": (0.5, 0.5)}

    w_news, w_market = weights.get(regime, (0.5, 0.5))

    final_score = (w_news * news_risk) + (w_market * market_risk)
    return round(final_score, 2)