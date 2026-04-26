"""
feature_engineer.py
-------------------
Converts raw longitudinal (per-row) ICU data into one feature vector
per patient.  Features extracted per clinical column:

  * last       – last observed value  (most clinically relevant)
  * mean       – mean over all time steps
  * std        – standard deviation
  * min / max  – range
  * trend      – linear regression slope (rate of change)
  * miss_rate  – fraction of originally-missing time steps

Static features (Age, Gender, Unit1, Unit2, HospAdmTime) are taken as-is
from the last row.

Label: 1 if the patient EVER had SepsisLabel == 1, else 0.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import List


STATIC_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]


def _slope(series: pd.Series) -> float:
    """Linear regression slope across time steps."""
    y = series.dropna().values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    try:
        slope, *_ = linregress(x, y)
        return float(slope)
    except Exception:
        return 0.0


def _aggregate_patient(group: pd.DataFrame, clinical_cols: List[str]) -> dict:
    """
    Given one patient's time-series group, return a flat feature dict.
    """
    features = {}

    # ── Clinical time-series features ───────────────────────────────────────
    for col in clinical_cols:
        if col not in group.columns:
            continue
        s = group[col]
        miss_col = f"{col}_missing"

        features[f"{col}_last"]  = s.iloc[-1]
        features[f"{col}_mean"]  = s.mean()
        features[f"{col}_std"]   = s.std(ddof=0)
        features[f"{col}_min"]   = s.min()
        features[f"{col}_max"]   = s.max()
        features[f"{col}_trend"] = _slope(s)

        # missingness rate from the indicator flag (computed before imputation)
        if miss_col in group.columns:
            features[f"{col}_miss_rate"] = group[miss_col].mean()

    # ── Static / demographic features ────────────────────────────────────────
    for col in STATIC_COLS:
        if col in group.columns:
            features[col] = group[col].iloc[-1]

    # ── Time-in-ICU ──────────────────────────────────────────────────────────
    features["n_time_steps"]    = len(group)
    features["max_ICULOS"]      = group["ICULOS"].max()
    features["HospAdmTime_val"] = group["HospAdmTime"].iloc[-1] if "HospAdmTime" in group.columns else np.nan

    return features


def engineer_features(df: pd.DataFrame, clinical_cols: List[str]) -> pd.DataFrame:
    """
    Aggregate the time-series DataFrame into one row per patient.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed longitudinal DataFrame with 'patient_id' column.
    clinical_cols : list[str]
        List of clinical column names to aggregate.

    Returns
    -------
    pd.DataFrame
        One row per patient. Includes 'SepsisLabel' (patient-level).
    """
    print("[Feature Engineering] Aggregating per-patient features...")

    records = []
    for pid, group in df.groupby("patient_id", sort=False):
        feat = _aggregate_patient(group, clinical_cols)
        feat["patient_id"]   = pid
        feat["SepsisLabel"]  = int(group["SepsisLabel"].max())   # 1 if ever septic
        records.append(feat)

    result = pd.DataFrame(records)
    print(f"[Feature Engineering] Shape: {result.shape}  |  Sepsis rate: "
          f"{result['SepsisLabel'].mean():.2%}")
    return result


def get_X_y(feat_df: pd.DataFrame):
    """
    Split feature DataFrame into X (features) and y (labels).
    Drops patient_id and SepsisLabel from X.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    feature_names : list[str]
    """
    drop_cols = ["patient_id", "SepsisLabel"]
    X = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns])
    y = feat_df["SepsisLabel"]

    # Final safety net: drop any remaining NaN / Inf columns
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y, list(X.columns)
