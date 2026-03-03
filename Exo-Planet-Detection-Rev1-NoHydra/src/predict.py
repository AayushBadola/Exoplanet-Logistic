"""
predict.py  -  Preprocess a raw sample and run inference with the saved model.
No Hydra / omegaconf.
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
def load_trained_model(cfg: dict):
    path = cfg["MODEL_PATH"]
    if not os.path.exists(path):
        logger.error(f"Model file not found: {path}")
        return None
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded <- {path}")
        return model
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}", exc_info=True)
        return None


def load_preprocessing_artifacts(cfg: dict) -> dict:
    artifacts: dict = {}
    name_path_pairs = [
        ("scaler",           cfg["SCALER_PATH"]),
        ("imputer",          cfg["IMPUTER_PATH"]),
        ("training_columns", cfg["TRAINING_COLUMNS_PATH"]),
    ]
    for name, path in name_path_pairs:
        if path and os.path.exists(path):
            try:
                artifacts[name] = joblib.load(path)
                logger.info(f"Loaded '{name}' <- {path}")
            except Exception as exc:
                logger.error(f"Failed to load '{name}': {exc}")
                artifacts[name] = None
        else:
            logger.warning(f"Artifact '{name}' not found at: {path}")
            artifacts[name] = None
    return artifacts


# -----------------------------------------------------------------------------
def preprocess_for_prediction(raw_df: pd.DataFrame,
                               artifacts: dict,
                               cfg: dict):
    """
    Apply the same transformations used during training to a raw sample.
    Returns a DataFrame aligned to training_columns, or None on failure.
    """
    if raw_df is None or raw_df.empty:
        logger.error("Input data for prediction is empty.")
        return None

    training_columns = artifacts.get("training_columns")
    scaler           = artifacts.get("scaler")
    imputer          = artifacts.get("imputer")

    if training_columns is None:
        logger.error("'training_columns' artifact missing.")
        return None

    # Strip column names
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    # Drop target if present
    target_col = cfg["TARGET_COLUMN"]
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Drop configured identifier columns
    drop_cols = [c for c in cfg["FEATURES_TO_DROP"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop object / string columns
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)

    # ------------------------------------------------------------------
    # Impute: must pass exactly the columns the imputer was fitted on
    # ------------------------------------------------------------------
    if imputer is not None:
        if hasattr(imputer, "feature_names_in_"):
            imputer_cols = list(imputer.feature_names_in_)
        else:
            imputer_cols = []

        if imputer_cols:
            # Add any columns the imputer expects but are missing in this sample
            for col in imputer_cols:
                if col not in df.columns:
                    df[col] = np.nan          # will be imputed to median
            # Run transform on exactly the fitted columns
            imputed_vals = imputer.transform(df[imputer_cols])
            for i, col in enumerate(imputer_cols):
                df[col] = imputed_vals[:, i]

    # ------------------------------------------------------------------
    # Ensure every training column exists; fill truly missing ones with 0
    # ------------------------------------------------------------------
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0.0
        elif df[col].isnull().any():
            df[col] = df[col].fillna(0.0)

    # ------------------------------------------------------------------
    # Scale: pass exactly the columns the scaler was fitted on
    # ------------------------------------------------------------------
    if scaler is not None:
        if hasattr(scaler, "feature_names_in_"):
            scale_cols = list(scaler.feature_names_in_)
        else:
            scale_cols = training_columns
        scale_cols = [c for c in scale_cols if c in df.columns]
        if scale_cols:
            df[scale_cols] = scaler.transform(df[scale_cols])

    # ------------------------------------------------------------------
    # Final alignment - exact columns in exact training order
    # ------------------------------------------------------------------
    final = df[training_columns].copy()

    nan_count = final.isnull().sum().sum()
    if nan_count > 0:
        logger.error(f"NaNs remain in prediction input ({nan_count} values).")
        return None

    logger.info(f"Prediction preprocessing complete - shape: {final.shape}")
    return final


# -----------------------------------------------------------------------------
def make_prediction(model, processed_df: pd.DataFrame):
    """Return (predictions_array, probabilities_array) or (None, None)."""
    if model is None or processed_df is None or processed_df.empty:
        return None, None
    try:
        preds  = model.predict(processed_df)
        probas = model.predict_proba(processed_df)
        return preds, probas
    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        return None, None
