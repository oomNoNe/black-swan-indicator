import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


class VIXForecaster:
    """Wrapper สำหรับการใช้ XGBoost ในการทำนายระดับ VIX ล่วงหน้า 7 วัน"""

    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4
        )
        self.is_trained = False
        self.train_score = None
        self.test_score = None

    def build_features(self, df):
        """สร้าง Lag Features ทางเศรษฐศาสตร์เพื่อป้อนให้โมเดล"""
        if df.empty or len(df) < 15:
            raise ValueError("Insufficient historical data for feature engineering.")

        data = df.copy()
        data['VIX_Lag1'] = data['VIX'].shift(1)
        data['VIX_Lag3'] = data['VIX'].shift(3)
        data['VIX_Lag7'] = data['VIX'].shift(7)
        data['SP500_Return_1D'] = data['Close'].pct_change(1)
        data['SP500_Return_5D'] = data['Close'].pct_change(5)

        # ตัวแปรพยากรณ์ล่วงหน้า 7 วัน
        data['Target_VIX_7D'] = data['VIX'].shift(-7)

        return data.dropna()

    def train_model(self, historical_data):
        """ดำเนินการ Train Model ย้อนหลัง"""
        try:
            df_features = self.build_features(historical_data)

            features = ['VIX_Lag1', 'VIX_Lag3', 'VIX_Lag7', 'SP500_Return_1D', 'SP500_Return_5D']
            X = df_features[features]
            y = df_features['Target_VIX_7D']

            # ไม่ใช้ shuffle เพื่อรักษารูปแบบ Time Series ไว้
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            self.model.fit(X_train, y_train)
            self.is_trained = True

            self.train_score = float(self.model.score(X_train, y_train))
            self.test_score = float(self.model.score(X_test, y_test))
            return {"status": "success", "r2_score": self.test_score}

        except Exception as e:
            print(f"[ML Training Error] XGBoost compilation failed: {e}")
            return {"status": "error"}

    def predict_vix(self, current_data):
        """ทำนายทิศทางและระดับราคาล่วงหน้า (Inference Wrapper)"""
        if not self.is_trained:
            return "Uninitialized"

        try:
            # ใช้แถวข้อมูลล่าสุดหลังจากสร้างฟีเจอร์เสร็จสิ้น
            df_features = self.build_features(current_data).iloc[-1:]
            X_pred = df_features[['VIX_Lag1', 'VIX_Lag3', 'VIX_Lag7', 'SP500_Return_1D', 'SP500_Return_5D']]

            prediction = self.model.predict(X_pred)[0]
            return float(round(float(prediction), 2))
        except Exception as e:
            print(f"[ML Prediction Error] Failed to forecast VIX: {e}")
            return np.nan