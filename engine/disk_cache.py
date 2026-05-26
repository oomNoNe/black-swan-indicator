"""
Disk-based cache สำหรับ persist trained models + DataFrames ระหว่าง session

ทำให้:
- รัน build_report.py ครั้งที่ 2 เร็วขึ้น 5-10x
- Streamlit restart ไม่ต้อง retrain ทุกครั้ง
- GitHub Actions ก็ได้ประโยชน์ (ถ้าเรา cache mlruns ด้วย)

ใช้ joblib สำหรับ sklearn-compat models, parquet สำหรับ DataFrames
"""
import os
import time
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable, Any

import pandas as pd

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _key_hash(key: str) -> str:
    """แปลง key เป็น short hash สำหรับ filename"""
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _cache_path(key: str, ext: str) -> Path:
    return CACHE_DIR / f"{_key_hash(key)}.{ext}"


def _is_fresh(path: Path, ttl_seconds: int) -> bool:
    """ไฟล์อายุยังไม่เกิน TTL?"""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_seconds


# ==========================================================
# DATAFRAME CACHE (parquet — compact + fast)
# ==========================================================
def cache_dataframe(
    key: str,
    fetch_fn: Callable[[], Optional[pd.DataFrame]],
    ttl_hours: float = 1.0
) -> Optional[pd.DataFrame]:
    """
    Cache DataFrame to disk as parquet

    Args:
        key: unique cache key (เช่น "macro_5y")
        fetch_fn: function ที่จะรันถ้า cache miss/expired
        ttl_hours: cache lifetime
    """
    path = _cache_path(key, "parquet")
    ttl = int(ttl_hours * 3600)

    if _is_fresh(path, ttl):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            print(f"[disk_cache] Failed to read {path}: {e} — refreshing")

    df = fetch_fn()
    if df is not None and not df.empty:
        try:
            df.to_parquet(path)
        except Exception as e:
            print(f"[disk_cache] Failed to write {path}: {e}")
    return df


# ==========================================================
# MODEL CACHE (joblib)
# ==========================================================
def cache_model(
    key: str,
    train_fn: Callable[[], Any],
    ttl_hours: float = 24.0
) -> Any:
    """
    Cache trained model to disk (joblib)

    Args:
        key: unique cache key (เช่น "xgboost_vix_5y")
        train_fn: function ที่จะรัน training ถ้า cache miss
        ttl_hours: cache lifetime (default 24h — VIX model OK ที่ใช้ 1 วัน)
    """
    if not JOBLIB_AVAILABLE:
        return train_fn()

    path = _cache_path(key, "joblib")
    ttl = int(ttl_hours * 3600)

    if _is_fresh(path, ttl):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"[disk_cache] Failed to load model {path}: {e}")

    model = train_fn()
    if model is not None:
        try:
            joblib.dump(model, path)
        except Exception as e:
            print(f"[disk_cache] Failed to save model {path}: {e}")
    return model


# ==========================================================
# PICKLE CACHE (สำหรับ dict, custom objects)
# ==========================================================
def cache_pickle(
    key: str,
    compute_fn: Callable[[], Any],
    ttl_hours: float = 24.0
) -> Any:
    """Cache arbitrary object ด้วย pickle"""
    path = _cache_path(key, "pkl")
    ttl = int(ttl_hours * 3600)

    if _is_fresh(path, ttl):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[disk_cache] Failed to load {path}: {e}")

    obj = compute_fn()
    if obj is not None:
        try:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        except Exception as e:
            print(f"[disk_cache] Failed to save {path}: {e}")
    return obj


# ==========================================================
# UTILS
# ==========================================================
def clear_cache():
    """ลบ cache ทั้งหมด"""
    count = 0
    for f in CACHE_DIR.glob("*"):
        if f.is_file():
            f.unlink()
            count += 1
    return count


def cache_info() -> dict:
    """ดู cache stats"""
    files = list(CACHE_DIR.glob("*"))
    return {
        "n_files": len(files),
        "total_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
        "files": [{"name": f.name, "size_kb": f.stat().st_size / 1024,
                   "age_min": (time.time() - f.stat().st_mtime) / 60}
                  for f in files],
    }
