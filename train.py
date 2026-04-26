"""
train.py
--------
Full training pipeline:
  1. Load data
  2. Preprocess (forward-fill, impute, missingness flags)
  3. Engineer features (per-patient aggregation)
  4. Train models (LR baseline + Random Forest)
  5. Evaluate and save metrics / plots
  6. Persist trained artefacts via joblib

Usage:
  python train.py --data_root data --model random_forest
"""

import os
import sys
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_loader      import load_all_data
from preprocessor     import preprocess, get_clinical_cols
from feature_engineer import engineer_features, get_X_y
from model            import get_model
from evaluator        import evaluate, plot_feature_importance


SEED = 42
np.random.seed(SEED)

MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sepsis Prediction Training Pipeline")
    p.add_argument("--data_root", default="data",
                   help="Root data folder (must contain Train_Data/ and Test_Data/)")
    p.add_argument("--model", default="random_forest",
                   choices=["logistic", "random_forest", "gradient_boosting"],
                   help="Model type to train")
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Number of cross-validation folds (0 = skip CV)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    train_raw, test_raw = load_all_data(args.data_root)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    train_proc, test_proc, medians = preprocess(train_raw, test_raw)
    clinical_cols = get_clinical_cols(train_proc)

    # Persist medians (needed for single-patient inference in GUI)
    joblib.dump(medians, os.path.join(MODEL_DIR, "medians.joblib"))

    # ── 3. Feature engineering ────────────────────────────────────────────────
    train_feat = engineer_features(train_proc, clinical_cols)
    test_feat  = engineer_features(test_proc,  clinical_cols)

    X_train, y_train, feature_names = get_X_y(train_feat)
    X_test,  y_test,  _             = get_X_y(test_feat)

    # Align columns (test may have extra/missing after engineering)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    # Persist feature names (needed for GUI inference)
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(clinical_cols, os.path.join(MODEL_DIR, "clinical_cols.joblib"))

    print(f"\nTrain patients: {len(X_train)} | Test patients: {len(X_test)}")
    print(f"Features: {len(feature_names)}")
    print(f"Train sepsis rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # ── 4. Baseline — Logistic Regression ─────────────────────────────────────
    print("\n--- Training Logistic Regression Baseline ---")
    lr_model = get_model("logistic")
    lr_model.fit(X_train, y_train)
    evaluate(lr_model, X_test, y_test, model_name="logistic_regression")

    # ── 5. Primary model ──────────────────────────────────────────────────────
    print(f"\n--- Training {args.model} ---")
    model = get_model(args.model)

    # Optional: cross-validation on training set
    if args.cv_folds > 1:
        print(f"Running {args.cv_folds}-fold CV...")
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
        cv_model = get_model(args.model)
        cv_model.fit(X_train, y_train)
        cv_f1 = cross_val_score(
            cv_model.pipeline, X_train, y_train,
            cv=skf, scoring="f1", n_jobs=-1,
        )
        print(f"CV F1 scores: {cv_f1.round(4)} | Mean: {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
        model = get_model(args.model)  # fresh model for final fit

    model.fit(X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate(model, X_test, y_test, model_name=args.model)

    # ── 7. Feature importance ─────────────────────────────────────────────────
    try:
        imp_df = model.feature_importances(feature_names)
        imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)
        plot_feature_importance(imp_df, args.model)
        print("\nTop 10 features:")
        print(imp_df.head(10).to_string(index=False))
    except NotImplementedError:
        pass

    # ── 8. Save model & metrics ───────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, f"{args.model}.joblib")
    joblib.dump(model, model_path)
    print(f"\n[Saved] Model → {model_path}")

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    def _to_python(v):
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)):    return int(v)
        return v
    with open(metrics_path, "w") as f:
        json.dump({k: _to_python(v) for k, v in metrics.items()}, f, indent=2)
    print(f"[Saved] Metrics → {metrics_path}")

    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
