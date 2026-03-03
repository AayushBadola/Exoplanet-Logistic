"""
config.py  –  All project settings in one plain Python file.
No Hydra, no omegaconf.  Works on Python 3.11, 3.12, 3.13, 3.14+.

Edit the values below to customise the pipeline.
"""
import os

# ── Resolve the project root regardless of where you run the script from ───────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Directory layout ───────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

# ── Dataset ────────────────────────────────────────────────────────────────────
RAW_DATA_FILE = os.path.join(DATA_DIR, "cumulative.csv")

# ── Target / label mapping ─────────────────────────────────────────────────────
TARGET_COLUMN  = "koi_disposition"
POSITIVE_LABELS = ["CONFIRMED", "CANDIDATE"]   # → class 1  (Exoplanet)
NEGATIVE_LABEL  = "FALSE POSITIVE"             # → class 0  (Not Exoplanet)

# ── Columns to drop before modelling ──────────────────────────────────────────
FEATURES_TO_DROP = [
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_pdisposition",
    "koi_score",           # pre-computed score → data leakage
    "koi_tce_delivname",
    "koi_comment",
    "koi_sparprov",
]

# ── Train / test split ─────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Logistic Regression hyperparameters ───────────────────────────────────────
MODEL_NAME   = "LogisticRegression"
MODEL_PARAMS = {
    "max_iter":     2000,
    "solver":       "saga",       # fast solver, works well for large datasets
    "C":            1.0,          # inverse regularisation strength (smaller = stronger)
    "class_weight": "balanced",   # handles class imbalance automatically
    "tol":          1e-4,
    "random_state": RANDOM_STATE,
}

# ── Saved-artifact paths ───────────────────────────────────────────────────────
MODEL_PATH            = os.path.join(MODEL_DIR, "logisticregression_model.joblib")
SCALER_PATH           = os.path.join(MODEL_DIR, "scaler.joblib")
IMPUTER_PATH          = os.path.join(MODEL_DIR, "imputer.joblib")
TRAINING_COLUMNS_PATH = os.path.join(MODEL_DIR, "training_columns.joblib")

# ── Prediction demo ────────────────────────────────────────────────────────────
PREDICTION_SAMPLE_SIZE = 10
RUN_PREDICTION_DEMO    = True

# ── Create directories if they don't exist ────────────────────────────────────
for _d in [DATA_DIR, MODEL_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)
