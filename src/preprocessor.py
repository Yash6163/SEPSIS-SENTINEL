"""
preprocessor.py
---------------
Time-series aware preprocessing pipeline:
  1. Forward-fill per patient (propagate last observed value)
  2. Backward-fill per patient (fill remaining NaNs from the next value)
  3. Global median imputation for any remaining NaNs
  4. Missingness indicator flags for every clinical feature
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# Columns that are NOT clinical measurements
NON_FEATURE_COLS = {"patient_id", "SepsisLabel", "ICULOS"}

# Static / demographic columns (don't forward-fill these)
STATIC_COLS = {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime"}


def get_clinical_cols(df: pd.DataFrame) -> List[str]:
    """Return list of clinical (numeric) columns to engineer features from."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c not in STATIC_COLS
    ]


def add_missingness_indicators(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    For each column in `cols`, add a binary flag column _missing
    that is 1 where the original value was NaN, 0 otherwise.
    Done BEFORE imputation so the flags reflect true missingness.
    """
    for col in cols:
        df[f"{col}_missing"] = df[col].isna().astype(np.int8)
    return df


def forward_backward_fill(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Apply per-patient forward-fill then backward-fill on clinical columns.
    This respects temporal ordering (ICULOS).
    """
    df = df.sort_values(["patient_id", "ICULOS"])
    df[cols] = (
        df.groupby("patient_id")[cols]
          .transform(lambda g: g.ffill().bfill())
    )
    return df


def global_median_impute(df: pd.DataFrame, cols: List[str], medians: dict = None):
    """
    Fill any remaining NaNs with the global median of each column.
    Pass `medians` dict when transforming test data (use train medians).

    Returns
    -------
    df : pd.DataFrame  (imputed)
    medians : dict     (column -> median value, useful to persist)
    """
    if medians is None:
        medians = {}
        for col in cols:
            medians[col] = df[col].median()

    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(medians.get(col, 0))

    return df, medians


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full preprocessing pipeline applied to train and test DataFrames.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Raw DataFrames as produced by data_loader.

    Returns
    -------
    train_processed, test_processed : pd.DataFrame
    medians : dict   (train medians used for imputation — save for inference)
    """
    clinical_cols = get_clinical_cols(train_df)

    print("[Preprocess] Adding missingness indicators...")
    train_df = add_missingness_indicators(train_df.copy(), clinical_cols)
    test_df  = add_missingness_indicators(test_df.copy(),  clinical_cols)

    print("[Preprocess] Forward/backward filling per patient...")
    train_df = forward_backward_fill(train_df, clinical_cols)
    test_df  = forward_backward_fill(test_df,  clinical_cols)

    print("[Preprocess] Global median imputation...")
    train_df, medians = global_median_impute(train_df, clinical_cols)
    test_df,  _       = global_median_impute(test_df,  clinical_cols, medians)

    # Also fill static columns with their mode (train) / same mode
    for col in STATIC_COLS:
        if col in train_df.columns:
            mode_val = train_df[col].mode()[0]
            train_df[col] = train_df[col].fillna(mode_val)
            test_df[col]  = test_df[col].fillna(mode_val)

    print(f"[Preprocess] Done. Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    return train_df, test_df, medians
