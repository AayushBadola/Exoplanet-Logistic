# 🪐 Exoplanet Detection — Revision 1: Logistic Regression

Binary classification of Kepler Objects of Interest (KOIs) using the
[NASA Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results) dataset.

**No Hydra / no omegaconf — works on Python 3.11, 3.12, 3.13, 3.14+.**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
#    Download cumulative.csv from Kaggle and put it in:
#    Exo-Planet-Detection-Rev1/data/cumulative.csv

# 3. Run
python main.py
```

That's it. All output lands in `reports/` and `logs/`.

---

## Project Structure

```
Exo-Planet-Detection-Rev1/
├── config.py          ← ALL settings in one plain Python file
├── main.py            ← Entry point
├── requirements.txt
├── data/
│   └── cumulative.csv   ← place dataset here
├── logs/
│   └── pipeline.log
├── models/            ← saved model + scaler/imputer/columns
└── src/
    ├── data_loader.py
    ├── preprocessor.py
    ├── model_trainer.py
    ├── predict.py
    └── logger_utils.py
```

---

## Plots Generated (`reports/`)

| File | Description |
|---|---|
| `confusion_matrix.png` | TP/FP/TN/FN with Sensitivity & Specificity |
| `roc_curve.png` | ROC curve + AUC fill |
| `pr_curve.png` | Precision-Recall curve + AUC fill |
| `metrics_summary.png` | Bar chart of 7 key metrics |
| `feature_coefficients.png` | Top 25 LR coefficients (blue=positive, orange=negative) |

---

## Metrics Computed

| Metric | Formula |
|---|---|
| Accuracy | (TP+TN) / total |
| Sensitivity (Recall) | TP / (TP+FN) |
| Specificity | TN / (TN+FP) |
| Precision | TP / (TP+FP) |
| F1 (Exoplanet class) | 2·P·R / (P+R) |
| ROC-AUC | Area under ROC curve |
| PR-AUC | Area under Precision-Recall curve |

---

## Configuration

All settings are in `config.py`. Key options:

```python
TEST_SIZE    = 0.20        # 80/20 split
RANDOM_STATE = 42

MODEL_PARAMS = {
    "max_iter":     2000,
    "solver":       "saga",
    "penalty":      "l2",
    "C":            1.0,           # smaller = stronger regularisation
    "class_weight": "balanced",    # handles class imbalance
}
```

---

## Preprocessing Steps

1. Strip whitespace from column names
2. Encode target: CONFIRMED/CANDIDATE → 1, FALSE POSITIVE → 0
3. Drop identifier & leakage columns (`koi_score`, `rowid`, etc.)
4. Drop remaining string/categorical columns
5. Drop entirely-NaN numeric columns
6. Median imputation for remaining NaNs
7. StandardScaler normalisation *(essential for Logistic Regression)*
8. Stratified 80/20 train-test split
