import yfinance as yf
import pandas as pd


# Macro tickers ที่ใช้เป็น features
# ----------------------------------
#   ^GSPC = S&P 500 (US equity benchmark)
#   ^VIX  = Volatility Index (fear gauge)
#   ^TNX  = 10-Year Treasury Yield (long-end of curve)
#   ^IRX  = 13-Week T-Bill Yield (short-end of curve)
#   GC=F  = Gold Futures (safe haven)
#   CL=F  = WTI Crude Oil Futures (growth/inflation proxy)
#   DX-Y.NYB = US Dollar Index (DXY) — global flight-to-quality
MACRO_TICKERS = {
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "TNX": "^TNX",      # 10Y yield
    "IRX": "^IRX",      # 3M yield (for yield curve)
    "Gold": "GC=F",
    "Oil": "CL=F",
    "DXY": "DX-Y.NYB",  # Dollar Index
}


# Multi-asset universe — Tier 3
# ทุกอันมี proxy สำหรับ "volatility/fear" เป็น Pair
ASSET_UNIVERSE = {
    # US Equity
    "S&P 500": {"price": "^GSPC", "vol": "^VIX", "category": "US Equity"},
    "Nasdaq 100": {"price": "^NDX", "vol": "^VXN", "category": "US Equity"},
    "Russell 2000": {"price": "^RUT", "vol": "^RVX", "category": "US Equity"},

    # International Equity
    "Emerging Markets (EEM)": {"price": "EEM", "vol": None, "category": "Emerging"},
    "China (FXI)": {"price": "FXI", "vol": None, "category": "Emerging"},
    "Brazil (EWZ)": {"price": "EWZ", "vol": None, "category": "Emerging"},

    # Crypto
    "Bitcoin": {"price": "BTC-USD", "vol": None, "category": "Crypto"},
    "Ethereum": {"price": "ETH-USD", "vol": None, "category": "Crypto"},

    # Commodities
    "Gold": {"price": "GC=F", "vol": None, "category": "Commodity"},
    "Oil (WTI)": {"price": "CL=F", "vol": "^OVX", "category": "Commodity"},
}


def get_current_vix():
    """ดึงค่า VIX ล่าสุด ณ ปัจจุบัน สำหรับแสดงบนหน้า Live Dashboard"""
    try:
        vix = yf.Ticker("^VIX")
        current_vix = vix.history(period="5d")['Close'].iloc[-1]
        return round(float(current_vix), 2)
    except Exception as e:
        print(f"[Market Crawler Error] Failed to get current VIX: {e}")
        return None


def fetch_historical_data(years=5):
    """
    ดึงข้อมูล S&P 500 และ VIX ย้อนหลัง — interface เดิม คงไว้สำหรับ backward compat
    (ใช้ใน regime_detector และ backtester ที่ต้องการแค่ Close + VIX)
    """
    try:
        sp500 = yf.Ticker("^GSPC").history(period=f"{years}y")
        vix = yf.Ticker("^VIX").history(period=f"{years}y")

        if sp500.empty or vix.empty:
            raise ValueError("Yahoo Finance returned empty data.")

        # ล้างโซนเวลาให้ตรงกัน
        sp500.index = pd.to_datetime(sp500.index).tz_localize(None).normalize()
        vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()

        df = pd.DataFrame()
        df['Close'] = sp500['Close']
        df['VIX'] = vix['Close']
        df = df.dropna()

        return df if not df.empty else None

    except Exception as e:
        print(f"[Market Crawler Error] Failed to fetch historical data: {e}")
        return None


def fetch_macro_data(years=5):
    """
    ดึงข้อมูล macroeconomic features สำหรับ ML model — ใหม่ใน Tier 2

    Returns DataFrame ที่มีคอลัมน์:
    - Close (S&P 500)
    - VIX
    - TNX (10Y Treasury yield)
    - IRX (3M T-Bill yield)
    - Gold, Oil, DXY
    - YieldCurve_Spread (TNX - IRX) — สัญญาณ recession ที่นิยมใช้

    ถ้า ticker ไหนดึงไม่ได้ → drop คอลัมน์นั้น แต่ยังคืน DataFrame
    """
    series = {}
    period = f"{years}y"

    for name, ticker in MACRO_TICKERS.items():
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                print(f"[Macro] {name} ({ticker}) returned empty — skipping")
                continue
            s = df['Close']
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            series[name] = s
        except Exception as e:
            print(f"[Macro] Failed to fetch {name} ({ticker}): {e} — skipping")

    if not series:
        return None

    df = pd.DataFrame(series)

    # Rename SP500 → Close สำหรับ backward compat
    if 'SP500' in df.columns:
        df = df.rename(columns={'SP500': 'Close'})

    # Yield curve spread: 10Y - 3M
    # ถ้าติดลบ = inverted curve = สัญญาณ recession (ทำนาย US recession ได้แม่นมากในอดีต)
    if 'TNX' in df.columns and 'IRX' in df.columns:
        df['YieldCurve_Spread'] = df['TNX'] - df['IRX']

    # Forward-fill ไม่เกิน 3 วัน (เผื่อกรณีตลาดบางแห่งปิด)
    df = df.ffill(limit=3).dropna(subset=['Close', 'VIX'])

    return df if not df.empty else None


def fetch_asset_data(asset_name, years=5):
    """
    Tier 3: ดึงข้อมูล asset อื่นๆ (crypto, EM, commodity)

    Args:
        asset_name: key จาก ASSET_UNIVERSE
        years: ระยะเวลาย้อนหลัง

    Returns:
        DataFrame with 'Close' and optionally 'VIX' (volatility proxy)
        คืน None ถ้าดึงไม่ได้
    """
    if asset_name not in ASSET_UNIVERSE:
        print(f"[Asset Fetch] Unknown asset: {asset_name}")
        return None

    config = ASSET_UNIVERSE[asset_name]
    period = f"{years}y"
    df = pd.DataFrame()

    try:
        # ดึงราคา
        price = yf.Ticker(config['price']).history(period=period)
        if price.empty:
            return None
        df['Close'] = price['Close']
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

        # ดึง volatility index ถ้ามี ไม่งั้นใช้ realized vol
        if config['vol']:
            vol = yf.Ticker(config['vol']).history(period=period)
            if not vol.empty:
                vol.index = pd.to_datetime(vol.index).tz_localize(None).normalize()
                df['VIX'] = vol['Close']
            else:
                df['VIX'] = _compute_realized_vol_proxy(df['Close'])
        else:
            # ไม่มี vol index → ใช้ realized vol 20 วันเป็น proxy (scale to VIX-like range)
            df['VIX'] = _compute_realized_vol_proxy(df['Close'])

        df = df.dropna()
        return df if not df.empty else None

    except Exception as e:
        print(f"[Asset Fetch Error] {asset_name}: {e}")
        return None


def _compute_realized_vol_proxy(close_series):
    """
    คำนวณ realized vol 20 วัน annualized × 100 → ใช้แทน VIX สำหรับ asset
    ที่ไม่มี dedicated volatility index (e.g. crypto, EM equity)
    """
    import numpy as np
    return close_series.pct_change().rolling(20).std() * np.sqrt(252) * 100
