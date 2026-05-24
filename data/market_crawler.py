import yfinance as yf
import pandas as pd


def get_current_vix():
    """ดึงค่า VIX ล่าสุด ณ ปัจจุบัน สำหรับแสดงบนหน้า Live Dashboard"""
    try:
        vix = yf.Ticker("^VIX")
        current_vix = vix.history(period="1d")['Close'].iloc[-1]
        return round(float(current_vix), 2)
    except Exception as e:
        print(f"[Market Crawler Error] Failed to get current VIX: {e}")
        return None


def fetch_historical_data(years=5):
    """ดึงข้อมูล S&P 500 และ VIX ย้อนหลังตามจำนวนปีที่กำหนด"""
    try:
        sp500 = yf.Ticker("^GSPC").history(period=f"{years}y")
        vix = yf.Ticker("^VIX").history(period=f"{years}y")

        if sp500.empty or vix.empty:
            raise ValueError("Yahoo Finance returned empty data.")

        # 🌟 THE MAGIC FIX: ล้างโซนเวลาและปรับเวลาให้เป็น 00:00:00 ตรงกัน
        # เพื่อบังคับให้ Pandas ประกบตาราง 2 ดัชนีได้สนิท 100%
        sp500.index = pd.to_datetime(sp500.index).tz_localize(None).normalize()
        vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()

        df = pd.DataFrame()
        df['Close'] = sp500['Close']
        df['VIX'] = vix['Close']

        # ลบแถวที่ข้อมูลไม่ครบออก
        df = df.dropna()

        if df.empty:
            return None

        return df

    except Exception as e:
        print(f"[Market Crawler Error] Failed to fetch historical data: {e}")
        return None