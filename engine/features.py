"""
Feature engineering สำหรับ ML pipeline

แยกออกมาจาก ml_predictor เพื่อให้:
1. Reusable across models (XGBoost, LightGBM, LSTM, Ridge)
2. Testable แยกชั้น
3. เพิ่ม feature ใหม่ได้ง่าย
"""
import pandas as pd
import numpy as np


# ขั้นต่ำของแถวที่ต้องมีหลัง dropna
MIN_ROWS_AFTER_FEATURES = 30


def build_features(df, target_horizon=7, classification=False, crash_threshold_pct=20.0):
    """
    สร้าง features + target จาก raw price data

    Args:
        df: DataFrame with at least 'Close' + 'VIX' (อาจมี macro cols เพิ่มเติม)
        target_horizon: ทำนายล่วงหน้ากี่วัน (default 7)
        classification: True = binary classification (crash vs no-crash),
                        False = regression (predict VIX level)
        crash_threshold_pct: % เพิ่มของ VIX ที่นับเป็น crash (default 20% ขึ้นใน 7 วัน)

    Returns:
        feature_df, feature_cols, target_col
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    if len(df) < 60:
        raise ValueError(f"Need at least 60 rows, got {len(df)}")

    data = df.copy()

    # ------------------------------------------------------
    # 1. VIX Lag features (เดิม)
    # ------------------------------------------------------
    data['VIX_Lag1'] = data['VIX'].shift(1)
    data['VIX_Lag3'] = data['VIX'].shift(3)
    data['VIX_Lag7'] = data['VIX'].shift(7)

    # VIX momentum features (ใหม่)
    data['VIX_Change_5D'] = data['VIX'].pct_change(5)
    data['VIX_MA_20'] = data['VIX'].rolling(20).mean()
    data['VIX_MA_Ratio'] = data['VIX'] / data['VIX_MA_20']  # VIX vs 20-day average

    # ------------------------------------------------------
    # 2. S&P 500 features
    # ------------------------------------------------------
    data['SP500_Return_1D'] = data['Close'].pct_change(1)
    data['SP500_Return_5D'] = data['Close'].pct_change(5)
    data['SP500_Vol_20D'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)

    # ------------------------------------------------------
    # 3. Macro features (ถ้ามี — ใหม่ใน Tier 2)
    # ------------------------------------------------------
    macro_features = []

    if 'YieldCurve_Spread' in data.columns:
        # Inverted yield curve = recession warning
        data['YC_Spread'] = data['YieldCurve_Spread']
        data['YC_Inverted'] = (data['YieldCurve_Spread'] < 0).astype(int)
        macro_features.extend(['YC_Spread', 'YC_Inverted'])

    if 'Gold' in data.columns:
        data['Gold_Return_5D'] = data['Gold'].pct_change(5)
        macro_features.append('Gold_Return_5D')

    if 'Oil' in data.columns:
        data['Oil_Return_5D'] = data['Oil'].pct_change(5)
        macro_features.append('Oil_Return_5D')

    if 'DXY' in data.columns:
        data['DXY_Return_5D'] = data['DXY'].pct_change(5)
        macro_features.append('DXY_Return_5D')

    # ------------------------------------------------------
    # 4. Target variable
    # ------------------------------------------------------
    if classification:
        # Binary: VIX จะเพิ่ม > X% ใน N วันข้างหน้ามั้ย?
        future_vix = data['VIX'].shift(-target_horizon)
        vix_pct_change = (future_vix - data['VIX']) / data['VIX'] * 100
        data['Target'] = (vix_pct_change > crash_threshold_pct).astype(int)
        target_col = 'Target'
    else:
        # Regression: ทำนายค่า VIX ในอนาคต
        data['Target_VIX_7D'] = data['VIX'].shift(-target_horizon)
        target_col = 'Target_VIX_7D'

    # ------------------------------------------------------
    # 5. รวม feature columns
    # ------------------------------------------------------
    core_features = [
        'VIX_Lag1', 'VIX_Lag3', 'VIX_Lag7',
        'VIX_Change_5D', 'VIX_MA_Ratio',
        'SP500_Return_1D', 'SP500_Return_5D', 'SP500_Vol_20D',
    ]
    feature_cols = core_features + macro_features

    # Drop rows with NaN ใน features หรือ target
    clean = data[feature_cols + [target_col]].dropna()

    if len(clean) < MIN_ROWS_AFTER_FEATURES:
        raise ValueError(f"After feature engineering, only {len(clean)} rows remain (need {MIN_ROWS_AFTER_FEATURES})")

    return clean, feature_cols, target_col
