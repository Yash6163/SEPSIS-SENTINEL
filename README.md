# 🏥 Sepsis Early Prediction System
### Healthcare AI Challenge — Round 3

---

## 📁 Folder Structure

```
sepsis_prediction/
│
├── data/                          # ← Place your dataset here
│   ├── Train_Data/                #   2000 patient PSV files (training)
│   │   ├── p100001.psv
│   │   ├── p100002.psv
│   │   └── ...
│   ├── Test_Data/                 #   2000 patient PSV files (testing)
│   │   ├── p200001.psv
│   │   └── ...
│   └── (all files also live here if using a flat combined folder)
│
├── src/                           # Core pipeline modules
│   ├── __init__.py
│   ├── data_loader.py             # PSV loading, patient assembly
│   ├── preprocessor.py            # Forward-fill, imputation, missingness flags
│   ├── feature_engineer.py        # Per-patient feature aggregation
│   ├── model.py                   # Model classes (LR, RF, GBM)
│   └── evaluator.py               # Metrics, confusion matrix, ROC curve
│
├── models/                        # Saved artefacts (created after training)
│   ├── random_forest.joblib       # Trained model
│   ├── medians.joblib             # Global medians for imputation
│   ├── feature_names.joblib       # Ordered feature list
│   └── clinical_cols.joblib       # Clinical column list
│
├── outputs/                       # Generated plots and metrics (after training)
│   ├── metrics.json
│   ├── feature_importances.csv
│   ├── roc_curve_random_forest.png
│   ├── confusion_matrix_random_forest.png
│   └── feature_importance_random_forest.png
│
├── train.py                       # ← Main training script
├── inference.py                   # Single-patient inference helper
├── app.py                         # ← Streamlit GUI
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Organise your data
```
data/
  Train_Data/   ← paste all training PSV files here
  Test_Data/    ← paste all test PSV files here
```

### Step 3 — Train the model
```bash
# Default (Random Forest, 5-fold CV)
python train.py --data_root data --model random_forest --cv_folds 5

# Logistic Regression baseline only
python train.py --data_root data --model logistic --cv_folds 0

# Gradient Boosting (stronger, slower)
python train.py --data_root data --model gradient_boosting --cv_folds 3
```

Training will print all metrics and save plots to `outputs/`.

### Step 4 — Launch the GUI
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Evaluation Metrics Produced

| Metric               | Where         |
|----------------------|---------------|
| Accuracy             | Terminal + JSON|
| Precision            | Terminal + JSON|
| Recall / Sensitivity | Terminal + JSON|
| F1-Score             | Terminal + JSON|
| Per-class Accuracy   | Terminal + JSON|
| ROC-AUC              | Plot + JSON   |
| Confusion Matrix     | Plot          |
| Inference Time       | Terminal + GUI|

---

## 🧠 Model Pipeline Summary

```
Raw PSV files
    │
    ▼
data_loader.py       — Reads all PSVs, adds patient_id
    │
    ▼
preprocessor.py      — Missingness flags → Forward/Backward fill → Median impute
    │
    ▼
feature_engineer.py  — Per patient: last / mean / std / min / max / trend / miss_rate
    │
    ▼
model.py             — RandomForest (balanced class weights, OOB scoring)
    │
    ▼
evaluator.py         — Full metrics suite + plots
    │
    ▼
inference.py + app.py — Real-time GUI predictions
```

---

## 🖥️ GUI Features

- **Manual entry** — key vitals + labs + demographics
- **PSV upload** — upload any patient file directly
- **Output displayed:**
  - Prediction (Sepsis / No Sepsis)
  - Probability percentage
  - Risk gauge (Low 🟢 / Medium 🟡 / High 🔴)
  - Inference time
  - Model metrics in sidebar
  - Feature importance bar chart
  - ROC Curve + Confusion Matrix images

---

## ⚠️ Constraints Compliance

| Constraint                            | Status |
|---------------------------------------|--------|
| No pretrained models                  | ✅ All models grown from scratch |
| No transfer learning                  | ✅ Pure sklearn estimators        |
| No external weights                   | ✅ Joblib saves only trained weights |
| Handle missing values                 | ✅ Forward-fill + imputation + flags |
| Handle class imbalance                | ✅ `class_weight="balanced"`      |
| Feature importance / interpretability | ✅ Built in                        |
| GUI mandatory                         | ✅ Streamlit app                   |
| All evaluation metrics                | ✅ evaluator.py                    |

---

## 🔁 Reproducibility

All random seeds are fixed at `SEED = 42` in `train.py`.
Run the same command twice → identical results.

---

## 📬 Tips for Best Results

- Use `--model gradient_boosting` for highest AUC (slower train)
- Use `--model random_forest` for best speed/accuracy tradeoff
- Recall is critical for sepsis — monitor it closely
- Check `outputs/feature_importances.csv` for top predictors


## MODEL DEMONSTRATION : 


https://github.com/user-attachments/assets/cd1a0913-f32a-47e1-ba8b-15b8bfb0ea90


