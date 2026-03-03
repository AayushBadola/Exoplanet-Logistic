"""
main.py  –  Revision 1: Logistic Regression
Kepler Exoplanet Detection Pipeline

Usage
-----
    python main.py

Requirements
------------
    pip install -r requirements.txt
    Place  data/cumulative.csv  (from Kaggle NASA dataset) in the data/ folder.
"""
import os
import sys

# ── Make sure the project root is on sys.path so `import config` always works ─
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd

# ── Project modules ────────────────────────────────────────────────────────────
import config                                       # plain-Python config dict
from src.logger_utils  import setup_logging
from src.data_loader   import load_data
from src.preprocessor  import preprocess_data, split_data
from src.model_trainer import train_model, evaluate_model, get_feature_importances
from src.predict       import preprocess_for_prediction, make_prediction

# ── Build the cfg dict that every module expects ──────────────────────────────
cfg = {
    # paths
    "DATA_DIR":              config.DATA_DIR,
    "MODEL_DIR":             config.MODEL_DIR,
    "REPORTS_DIR":           config.REPORTS_DIR,
    "LOGS_DIR":              config.LOGS_DIR,
    "RAW_DATA_FILE":         config.RAW_DATA_FILE,
    "MODEL_PATH":            config.MODEL_PATH,
    "SCALER_PATH":           config.SCALER_PATH,
    "IMPUTER_PATH":          config.IMPUTER_PATH,
    "TRAINING_COLUMNS_PATH": config.TRAINING_COLUMNS_PATH,
    # target
    "TARGET_COLUMN":         config.TARGET_COLUMN,
    "POSITIVE_LABELS":       config.POSITIVE_LABELS,
    "NEGATIVE_LABEL":        config.NEGATIVE_LABEL,
    # features
    "FEATURES_TO_DROP":      config.FEATURES_TO_DROP,
    # split
    "TEST_SIZE":             config.TEST_SIZE,
    "RANDOM_STATE":          config.RANDOM_STATE,
    # model
    "MODEL_NAME":            config.MODEL_NAME,
    "MODEL_PARAMS":          config.MODEL_PARAMS,
    # demo
    "PREDICTION_SAMPLE_SIZE": config.PREDICTION_SAMPLE_SIZE,
    "RUN_PREDICTION_DEMO":    config.RUN_PREDICTION_DEMO,
}

LOG_FILE = os.path.join(config.LOGS_DIR, "pipeline.log")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger = setup_logging(LOG_FILE)

    logger.info("=" * 65)
    logger.info("  Exoplanet Detection Pipeline  –  Revision 1: Logistic Regression")
    logger.info("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    raw_df = load_data(cfg["RAW_DATA_FILE"])
    if raw_df is None:
        logger.error("Data loading failed. Exiting.")
        sys.exit(1)
    logger.info(f"Raw data shape: {raw_df.shape}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    X, y, transformers = preprocess_data(raw_df, cfg, save_artifacts=True)
    if X is None or y is None:
        logger.error("Preprocessing failed. Exiting.")
        sys.exit(1)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)
    if X_train is None:
        logger.error("Train/test split failed. Exiting.")
        sys.exit(1)

    logger.info(f"Train samples : {len(X_train):,}  (pos={y_train.sum():,}  neg={(y_train==0).sum():,})")
    logger.info(f"Test  samples : {len(X_test):,}  (pos={y_test.sum():,}  neg={(y_test==0).sum():,})")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    model = train_model(X_train, y_train, cfg)
    if model is None:
        logger.error("Model training failed. Exiting.")
        sys.exit(1)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test, cfg)
    if not metrics:
        logger.error("Evaluation returned no metrics.")
        sys.exit(1)

    # ── 6. Feature coefficients ───────────────────────────────────────────────
    get_feature_importances(model, X_train.columns.tolist(), cfg, top_n=25)

    # ── 7. Prediction demo ────────────────────────────────────────────────────
    if cfg["RUN_PREDICTION_DEMO"]:
        _run_prediction_demo(model, raw_df, y_test, cfg, logger)

    logger.info("")
    logger.info("✓  Pipeline finished successfully.")
    logger.info(f"   Plots  → {cfg['REPORTS_DIR']}")
    logger.info(f"   Model  → {cfg['MODEL_PATH']}")
    logger.info(f"   Log    → {LOG_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
def _run_prediction_demo(model, raw_df, y_test, cfg, logger):
    logger.info("")
    logger.info("── Prediction Demo ──────────────────────────────────────────────")

    # Load saved artifacts
    artifacts = {}
    for name, path in [
        ("scaler",           cfg["SCALER_PATH"]),
        ("imputer",          cfg["IMPUTER_PATH"]),
        ("training_columns", cfg["TRAINING_COLUMNS_PATH"]),
    ]:
        if os.path.exists(path):
            artifacts[name] = joblib.load(path)
        else:
            logger.warning(f"Artifact '{name}' not found at {path} – skipping demo.")
            return

    n = min(cfg["PREDICTION_SAMPLE_SIZE"], len(y_test))

    # Sample row indices from the test set
    sample_y   = y_test.sample(n=n, random_state=cfg["RANDOM_STATE"])
    sample_idx = sample_y.index                    # integer positions (reset_index was called)

    # Pull the same rows out of the original raw_df by integer position
    sample_raw = raw_df.iloc[sample_idx].copy()

    # Build actual labels for display
    target_col  = cfg["TARGET_COLUMN"]
    pos_labels  = cfg["POSITIVE_LABELS"]
    neg_label   = cfg["NEGATIVE_LABEL"]

    def _to_binary(val):
        if val in pos_labels:  return 1
        if val == neg_label:   return 0
        return -1

    actual_labels = None
    if target_col in sample_raw.columns:
        actual_labels = sample_raw[target_col].apply(_to_binary).values

    # Preprocess & predict
    processed = preprocess_for_prediction(sample_raw, artifacts, cfg)
    if processed is None:
        logger.error("Prediction demo: preprocessing failed.")
        return

    preds, probas = make_prediction(model, processed)
    if preds is None:
        logger.error("Prediction demo: inference failed.")
        return

    # ── Pretty table (no external lib needed) ─────────────────────────────
    header = f"{'#':>3}  {'Actual':<16}  {'Predicted':<16}  {'Confidence':>10}  {'P(Exoplanet)':>12}"
    sep    = "─" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for i in range(len(preds)):
        act_str  = ("Exoplanet" if actual_labels[i] == 1 else "Not Exoplanet") \
                   if actual_labels is not None and actual_labels[i] != -1 else "Unknown"
        pred_str = "Exoplanet" if preds[i] == 1 else "Not Exoplanet"
        p1       = probas[i][1]
        conf     = probas[i][int(preds[i])]
        logger.info(f"{i+1:>3}  {act_str:<16}  {pred_str:<16}  {conf:>10.4f}  {p1:>12.4f}")

    logger.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
