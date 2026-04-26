"""
inference.py
------------
Single-patient inference pipeline.
Accepts raw time-series rows (as a DataFrame) for ONE patient,
runs preprocessing + feature engineering + model prediction,
and returns structured results.

Used by app.py (GUI) for real-time predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocessor     import get_clinical_cols, add_missingness_indicators, STATIC_COLS
from feature_engineer import engineer_features, get_X_y

MODEL_DIR = "models"


class SepsisPredictor:
    """
    Wraps the saved model + artefacts for single-patient inference.

    Usage
    -----
    predictor = SepsisPredictor()
    result    = predictor.predict(patient_df)
    """

    def __init__(self, model_name: str = "random_forest"):
        self.model_name = model_name
        self._load_artefacts()

    def _load_artefacts(self):
        model_path   = os.path.join(MODEL_DIR, f"{self.model_name}.joblib")
        medians_path = os.path.join(MODEL_DIR, "medians.joblib")
        featnames_path = os.path.join(MODEL_DIR, "feature_names.joblib")
        clincols_path  = os.path.join(MODEL_DIR, "clinical_cols.joblib")

        for p in [model_path, medians_path, featnames_path, clincols_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Artefact not found: {p}\n"
                    "Please run train.py first to generate model artefacts."
                )

        self.model         = joblib.load(model_path)
        self.medians       = joblib.load(medians_path)
        self.feature_names = joblib.load(featnames_path)
        self.clinical_cols = joblib.load(clincols_path)

    def predict(self, patient_df: pd.DataFrame) -> dict:
        """
        Predict sepsis risk for a single patient.

        Parameters
        ----------
        patient_df : pd.DataFrame
            Time-series rows for one patient.
            Must contain the same column names as the training data.
            'patient_id' column is added if missing.
            'SepsisLabel' column is optional.

        Returns
        -------
        dict with keys:
            prediction     : int (0 or 1)
            probability    : float (0–1)
            risk_level     : str ("Low" / "Medium" / "High")
            confidence_pct : float (0–100)
        """
        df = patient_df.copy()

        # Ensure required columns
        if "patient_id" not in df.columns:
            df.insert(0, "patient_id", "live_patient")
        if "SepsisLabel" not in df.columns:
            df["SepsisLabel"] = 0
        if "ICULOS" not in df.columns:
            df["ICULOS"] = range(1, len(df) + 1)

        # Missingness indicators
        df = add_missingness_indicators(df, self.clinical_cols)

        # Forward/backward fill
        df = df.sort_values("ICULOS")
        df[self.clinical_cols] = df[self.clinical_cols].ffill().bfill()

        # Median imputation
        for col in self.clinical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.medians.get(col, 0))
        for col in STATIC_COLS:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(0)

        # Feature engineering
        feat_df = engineer_features(df, self.clinical_cols)
        X, _, _ = get_X_y(feat_df)

        # Align feature columns
        X = X.reindex(columns=self.feature_names, fill_value=0)

        # Predict
        prediction  = int(self.model.predict(X)[0])
        probability = float(self.model.predict_proba(X)[0])

        risk_level = (
            "Low"    if probability < 0.35 else
            "Medium" if probability < 0.65 else
            "High"
        )

        return {
            "prediction":     prediction,
            "probability":    probability,
            "confidence_pct": round(probability * 100, 1),
            "risk_level":     risk_level,
        }
