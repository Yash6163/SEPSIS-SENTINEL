"""
evaluator.py
------------
Computes and displays all required evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - Per-class accuracy
  - Confusion matrix
  - ROC-AUC curve
  - Inference time
Saves plots to outputs/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             model_name: str = "model") -> dict:
    """
    Run full evaluation suite on the given model and test set.

    Returns
    -------
    dict of all metric values (suitable for logging / display).
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {model_name.upper()}")
    print(f"{'='*60}")

    # ── Inference ────────────────────────────────────────────────────────────
    t0     = time.perf_counter()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    inf_ms = (time.perf_counter() - t0) * 1000

    # ── Basic metrics ─────────────────────────────────────────────────────────
    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # ── Per-class accuracy ────────────────────────────────────────────────────
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc_class0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # specificity
    acc_class1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # sensitivity / recall

    print(f"\nAccuracy        : {acc:.4f}")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1-Score        : {f1:.4f}")
    print(f"ROC-AUC         : {roc_auc:.4f}")
    print(f"Per-class Acc   : Non-Sepsis={acc_class0:.4f} | Sepsis={acc_class1:.4f}")
    print(f"Inference time  : {inf_ms:.2f} ms (total for {len(X_test)} samples)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Non-Sepsis','Sepsis'])}")

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, model_name)
    plot_roc_curve(y_test, y_prob, roc_auc, model_name)

    return {
        "model":            model_name,
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "roc_auc":          roc_auc,
        "acc_non_sepsis":   acc_class0,
        "acc_sepsis":       acc_class1,
        "inference_ms":     inf_ms,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Sepsis", "Sepsis"],
        yticklabels=["Non-Sepsis", "Sepsis"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ROC Curve Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, roc_auc: float, model_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, top_n: int = 20):
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#2ca02c")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"feature_importance_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")
    return path
