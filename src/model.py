"""
model.py
--------
Model zoo — all built from scratch using scikit-learn base estimators ONLY
(no pretrained weights, no external model files).

Models available:
  1. LogisticRegressionModel  – fast baseline
  2. RandomForestModel        – main model (handles imbalance via class_weight)
  3. GradientBoostingModel    – stronger ensemble (pure sklearn, no XGBoost)

All models expose a uniform interface:
  .fit(X, y)
  .predict(X)
  .predict_proba(X)
  .feature_importances(feature_names)
"""

import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Base wrapper
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel:
    def __init__(self):
        self.pipeline = None
        self.feature_names_ = None
        self.inference_time_ms_ = None
        self.threshold = 0.35   # lower threshold boosts Recall for rare sepsis class

    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        t0 = time.perf_counter()
        proba = self.pipeline.predict_proba(X)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        self.inference_time_ms_ = (time.perf_counter() - t0) * 1000
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def feature_importances(self, feature_names=None) -> pd.DataFrame:
        """Return a DataFrame of feature importances (if available)."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Logistic Regression (baseline)
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRegressionModel(BaseModel):
    """
    L2-regularised logistic regression with class-weight balancing.
    Scales features internally via StandardScaler.
    """
    def __init__(self, C: float = 0.1, max_iter: int = 1000, random_state: int = 42):
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight="balanced",
                solver="lbfgs",
                random_state=self.random_state,
                n_jobs=-1,
            ))
        ])
        self.pipeline.fit(X, y)
        print(f"[LogisticRegression] Fitted on {len(X)} samples.")
        return self

    def feature_importances(self, feature_names=None) -> pd.DataFrame:
        names = feature_names or self.feature_names_
        coef  = self.pipeline.named_steps["clf"].coef_[0]
        return (
            pd.DataFrame({"feature": names, "importance": np.abs(coef)})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest  (primary model)
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel(BaseModel):
    """
    Random Forest with balanced class weights and bootstrap sampling.
    No external pretrained weights — trees are grown from scratch.
    """
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 12,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state    = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",   # rebalance per tree for better recall
            bootstrap=True,
            oob_score=True,
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.pipeline = Pipeline([("clf", clf)])
        self.pipeline.fit(X, y)
        oob = self.pipeline.named_steps["clf"].oob_score_
        print(f"[RandomForest] Fitted. OOB accuracy: {oob:.4f}")
        return self

    def feature_importances(self, feature_names=None) -> pd.DataFrame:
        names = feature_names or self.feature_names_
        imps  = self.pipeline.named_steps["clf"].feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": imps})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Boosting  (stronger, slower)
# ─────────────────────────────────────────────────────────────────────────────

class GradientBoostingModel(BaseModel):
    """
    sklearn GradientBoostingClassifier — pure decision-tree ensemble,
    no external libraries.  Uses sample_weight for class imbalance.
    """
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators   = n_estimators
        self.learning_rate  = learning_rate
        self.max_depth      = max_depth
        self.random_state   = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = list(X.columns)

        # Compute sample weights to handle class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        weight_pos = len(y) / (2.0 * n_pos) if n_pos > 0 else 1.0
        weight_neg = len(y) / (2.0 * n_neg) if n_neg > 0 else 1.0
        sample_weight = np.where(y == 1, weight_pos, weight_neg)

        clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            random_state=self.random_state,
        )
        self.pipeline = Pipeline([("clf", clf)])
        self.pipeline.fit(X, y, clf__sample_weight=sample_weight)
        print(f"[GradientBoosting] Fitted on {len(X)} samples.")
        return self

    def feature_importances(self, feature_names=None) -> pd.DataFrame:
        names = feature_names or self.feature_names_
        imps  = self.pipeline.named_steps["clf"].feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": imps})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_model(name: str = "random_forest") -> BaseModel:
    """
    Factory function.  name in {"logistic", "random_forest", "gradient_boosting"}
    """
    mapping = {
        "logistic":          LogisticRegressionModel,
        "random_forest":     RandomForestModel,
        "gradient_boosting": GradientBoostingModel,
    }
    if name not in mapping:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(mapping)}")
    return mapping[name]()
