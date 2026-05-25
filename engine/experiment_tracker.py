"""
MLflow Experiment Tracking — Tier 3

ใช้สำหรับบันทึก parameters, metrics, artifacts ของ ML experiments
ทำให้เปรียบเทียบ runs ข้ามเวลาได้ + reproducible

วิธีดู results:
1. รัน app → กดปุ่ม "Log experiment to MLflow"
2. เปิด terminal ใหม่: `mlflow ui --backend-store-uri ./mlruns`
3. เปิด http://localhost:5000

MLflow tracking URI ใช้ local file store (./mlruns/)
- ฟรี ไม่ต้อง server
- commit ขึ้น GitHub ได้ (เห็น history ของ experiments)
"""
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT_NAME = "black-swan-indicator"


def _ensure_setup():
    """ตั้งค่า MLflow ครั้งเดียว"""
    if not MLFLOW_AVAILABLE:
        return False
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT_NAME))
    return True


@contextmanager
def track_run(run_name=None, tags=None):
    """Context manager wrapper รอบ mlflow.start_run()"""
    if not _ensure_setup():
        yield None
        return

    run_name = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_walkforward_result(result, run_name=None):
    """
    บันทึกผล walk_forward_validate() ลง MLflow

    Args:
        result: dict จาก walk_forward_validate()
        run_name: optional, default = "{model}_{task}_{timestamp}"

    Returns:
        run_id (str) หรือ None ถ้า MLflow ไม่พร้อม
    """
    if not MLFLOW_AVAILABLE:
        return None

    model = result.get("model", "unknown")
    task = result.get("task", "unknown")
    run_name = run_name or f"{model}_{task}"

    with track_run(run_name=run_name, tags={"model": model, "task": task}) as run:
        if run is None:
            return None

        # Log parameters
        mlflow.log_params({
            "model": model,
            "task": task,
            "n_splits": result.get("n_splits"),
            "primary_metric": result.get("primary_metric"),
        })

        # Log aggregate metrics
        mlflow.log_metrics({
            "mean_score": result.get("mean_score", 0),
            "std_score": result.get("std_score", 0),
        })

        # Log per-fold metrics
        for fold in result.get("fold_scores", []):
            fold_idx = fold.get("fold", 0)
            for key, value in fold.items():
                if key == "fold" or not isinstance(value, (int, float)):
                    continue
                mlflow.log_metric(f"fold_{fold_idx}_{key}", float(value))

        return run.info.run_id


def log_model_comparison(comparison_df, task="regression"):
    """บันทึกผล compare_models() ลง MLflow (1 run รวม)"""
    if not MLFLOW_AVAILABLE or comparison_df is None or comparison_df.empty:
        return None

    run_name = f"comparison_{task}_{datetime.utcnow().strftime('%H%M%S')}"
    with track_run(run_name=run_name, tags={"task": task, "type": "comparison"}) as run:
        if run is None:
            return None
        mlflow.log_param("task", task)
        mlflow.log_param("n_models", len(comparison_df))

        for _, row in comparison_df.iterrows():
            model = row['Model']
            if "Mean Score" in row and isinstance(row['Mean Score'], (int, float)):
                mlflow.log_metric(f"{model}_mean", float(row['Mean Score']))
                mlflow.log_metric(f"{model}_std", float(row['Std Score']))

        # Save table as artifact
        csv_path = "/tmp/comparison.csv" if os.name != 'nt' else os.path.join(os.environ.get('TEMP', '.'), 'comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        return run.info.run_id


def get_recent_runs(max_results: int = 20) -> Optional[pd.DataFrame]:
    """
    ดึง runs ล่าสุดสำหรับแสดงใน UI

    Returns:
        DataFrame ของ runs (อาจเป็น empty) หรือ None ถ้า MLflow ไม่พร้อม
    """
    if not _ensure_setup():
        return None
    try:
        runs = mlflow.search_runs(
            order_by=["start_time DESC"],
            max_results=max_results,
            output_format="pandas",  # บังคับ DataFrame เสมอ
        )
        # ป้องกันเคส mlflow คืน list — แปลงเป็น DataFrame
        if not isinstance(runs, pd.DataFrame):
            return None
        return runs
    except Exception as e:
        print(f"[MLflow] Failed to load runs: {e}")
        return None
