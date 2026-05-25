"""
ML Predictor — รองรับหลายโมเดล (XGBoost, LightGBM, Ridge) ทั้ง regression และ classification
+ Walk-forward validation (best practice สำหรับ time-series)
+ SHAP feature importance
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from engine.features import build_features


# ==========================================================
# MODEL REGISTRY — เพิ่มโมเดลใหม่ทำได้ที่นี่
# ==========================================================
def _make_model(name, task):
    """Factory: สร้าง model instance ตามชื่อ + task type"""
    if task == "regression":
        if name == "XGBoost":
            return xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150,
                                    learning_rate=0.05, max_depth=4, verbosity=0)
        if name == "LightGBM":
            return lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05,
                                     max_depth=4, num_leaves=15, verbose=-1)
        if name == "Ridge":
            return Ridge(alpha=1.0)
    elif task == "classification":
        if name == "XGBoost":
            return xgb.XGBClassifier(n_estimators=150, learning_rate=0.05,
                                     max_depth=4, eval_metric='logloss', verbosity=0)
        if name == "LightGBM":
            return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                                      max_depth=4, num_leaves=15, verbose=-1)
        if name == "LogReg":
            return LogisticRegression(max_iter=1000, class_weight='balanced')
    raise ValueError(f"Unknown model {name} for task {task}")


SUPPORTED_MODELS = {
    "regression": ["XGBoost", "LightGBM", "Ridge"],
    "classification": ["XGBoost", "LightGBM", "LogReg"],
}


# ==========================================================
# VIX FORECASTER (Regression) — interface เดิม + ฟีเจอร์ใหม่
# ==========================================================
class VIXForecaster:
    """Regression wrapper — ทำนายค่า VIX 7 วันข้างหน้า"""

    def __init__(self, model_name="XGBoost"):
        self.model_name = model_name
        self.model = _make_model(model_name, "regression")
        self.is_trained = False
        self.train_score = None
        self.test_score = None
        self.feature_cols = None

    def build_features(self, df):
        """คงไว้สำหรับ backward compat — delegate ไปยัง engine.features"""
        feat_df, _, _ = build_features(df, classification=False)
        return feat_df

    def train_model(self, historical_data):
        """Train ครั้งเดียวแบบ 80/20 split (สำหรับ quick training)"""
        try:
            clean, feature_cols, target_col = build_features(historical_data, classification=False)
            self.feature_cols = feature_cols

            X = clean[feature_cols]
            y = clean[target_col]

            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            self.model.fit(X_train, y_train)
            self.is_trained = True

            self.train_score = float(r2_score(y_train, self.model.predict(X_train)))
            self.test_score = float(r2_score(y_test, self.model.predict(X_test)))
            return {"status": "success", "r2_score": self.test_score}

        except Exception as e:
            print(f"[ML Training Error] {self.model_name} failed: {e}")
            return {"status": "error", "error": str(e)}

    def predict_vix(self, current_data):
        """ทำนายค่า VIX 7 วันข้างหน้า"""
        if not self.is_trained:
            return "Uninitialized"

        try:
            clean, feature_cols, _ = build_features(current_data, classification=False)
            X_pred = clean[feature_cols].iloc[-1:]
            prediction = self.model.predict(X_pred)[0]
            return float(round(float(prediction), 2))
        except Exception as e:
            print(f"[ML Prediction Error] {e}")
            return np.nan


# ==========================================================
# WALK-FORWARD VALIDATION (ใหม่ใน Tier 2)
# ==========================================================
def walk_forward_validate(df, model_name="XGBoost", task="regression",
                          n_splits=5, classification_kwargs=None):
    """
    Walk-forward validation — มาตรฐาน time-series ML

    หลักการ: train บนข้อมูลช่วง 1, test ช่วง 2 → train 1+2, test 3 → ...
    ป้องกัน look-ahead bias ดีกว่า train/test split ธรรมดา

    Args:
        df: raw price data
        model_name: หนึ่งใน SUPPORTED_MODELS[task]
        task: 'regression' หรือ 'classification'
        n_splits: จำนวน fold
        classification_kwargs: dict สำหรับส่งต่อให้ build_features ถ้า task='classification'

    Returns:
        dict with 'mean_score', 'std_score', 'fold_scores', 'predictions_df'
    """
    is_classification = (task == "classification")
    kwargs = classification_kwargs or {}

    clean, feature_cols, target_col = build_features(
        df, classification=is_classification, **kwargs
    )

    X = clean[feature_cols]
    y = clean[target_col].values
    dates = clean.index

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    all_preds = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = _make_model(model_name, task)
        X_train_df, X_test_df = X.iloc[train_idx], X.iloc[test_idx]
        model.fit(X_train_df, y[train_idx])
        y_pred = model.predict(X_test_df)

        if is_classification:
            y_proba = model.predict_proba(X_test_df)[:, 1] if hasattr(model, "predict_proba") else None
            fold_results.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": float(accuracy_score(y[test_idx], y_pred)),
                "precision": float(precision_score(y[test_idx], y_pred, zero_division=0)),
                "recall": float(recall_score(y[test_idx], y_pred, zero_division=0)),
                "f1": float(f1_score(y[test_idx], y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y[test_idx], y_proba)) if y_proba is not None and len(set(y[test_idx])) > 1 else np.nan,
            })
        else:
            fold_results.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "r2": float(r2_score(y[test_idx], y_pred)),
                "mae": float(mean_absolute_error(y[test_idx], y_pred)),
            })

        for date, true_val, pred_val in zip(dates[test_idx], y[test_idx], y_pred):
            all_preds.append({"date": date, "fold": fold_idx,
                              "actual": float(true_val), "predicted": float(pred_val)})

    primary_metric = "f1" if is_classification else "r2"
    scores = [f[primary_metric] for f in fold_results]

    return {
        "model": model_name,
        "task": task,
        "n_splits": n_splits,
        "primary_metric": primary_metric,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "fold_scores": fold_results,
        "predictions_df": pd.DataFrame(all_preds),
    }


# ==========================================================
# MODEL COMPARISON (ใหม่ใน Tier 2)
# ==========================================================
def compare_models(df, task="regression", n_splits=5, classification_kwargs=None):
    """
    เทียบหลายโมเดลด้วย walk-forward validation
    คืน DataFrame เรียงตาม performance
    """
    results = []
    for model_name in SUPPORTED_MODELS[task]:
        try:
            res = walk_forward_validate(df, model_name=model_name, task=task,
                                        n_splits=n_splits,
                                        classification_kwargs=classification_kwargs)
            results.append({
                "Model": model_name,
                "Mean Score": round(res["mean_score"], 4),
                "Std Score": round(res["std_score"], 4),
                "Metric": res["primary_metric"].upper(),
            })
        except Exception as e:
            print(f"[Compare] {model_name} failed: {e}")
            results.append({"Model": model_name, "Mean Score": np.nan,
                            "Std Score": np.nan, "Metric": "ERROR"})

    df_results = pd.DataFrame(results).sort_values("Mean Score", ascending=False).reset_index(drop=True)
    return df_results


# ==========================================================
# CLASSIFICATION (Crash Detector) — ใหม่ใน Tier 2
# ==========================================================
class CrashClassifier:
    """Binary classifier: VIX จะ spike > X% ใน N วันมั้ย?"""

    def __init__(self, model_name="XGBoost", crash_threshold_pct=20.0, target_horizon=7):
        self.model_name = model_name
        self.crash_threshold_pct = crash_threshold_pct
        self.target_horizon = target_horizon
        self.model = _make_model(model_name, "classification")
        self.is_trained = False
        self.metrics = None
        self.feature_cols = None

    def train(self, historical_data):
        try:
            clean, feature_cols, target_col = build_features(
                historical_data, classification=True,
                crash_threshold_pct=self.crash_threshold_pct,
                target_horizon=self.target_horizon,
            )
            self.feature_cols = feature_cols

            X = clean[feature_cols]
            y = clean[target_col]

            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            self.model.fit(X_train, y_train)
            self.is_trained = True

            y_pred = self.model.predict(X_test)
            self.metrics = {
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "F1": float(f1_score(y_test, y_pred, zero_division=0)),
                "Base Rate (crash %)": float(y_test.mean() * 100),
            }
            return {"status": "success", "metrics": self.metrics}
        except Exception as e:
            print(f"[CrashClassifier Error] {e}")
            return {"status": "error", "error": str(e)}

    def predict_crash_probability(self, current_data):
        """คืน probability (0-1) ว่าจะเกิด crash ใน N วันข้างหน้า"""
        if not self.is_trained:
            return None
        try:
            clean, feature_cols, _ = build_features(
                current_data, classification=True,
                crash_threshold_pct=self.crash_threshold_pct,
                target_horizon=self.target_horizon,
            )
            X_pred = clean[feature_cols].iloc[-1:]
            return float(self.model.predict_proba(X_pred)[0, 1])
        except Exception as e:
            print(f"[CrashClassifier Predict Error] {e}")
            return None


# ==========================================================
# SHAP ANALYSIS (ใหม่ใน Tier 2)
# ==========================================================
def compute_shap_values(model, X_sample, max_samples=200):
    """
    คำนวณ SHAP values สำหรับ tree-based models
    คืน (shap_values, feature_names) หรือ None ถ้า error
    """
    try:
        import shap
        # จำกัด sample เพื่อความเร็ว (SHAP บน tree เร็ว แต่ก็ไม่อยากให้ user รอ)
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(max_samples, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # XGBoost classifier บางครั้งคืน list[array] (one per class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        return shap_values, list(X_sample.columns), X_sample
    except Exception as e:
        print(f"[SHAP Error] {e}")
        return None, None, None
