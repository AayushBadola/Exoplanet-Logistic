"""
preprocessor.py
Steps
  1. Strip column-name whitespace
  2. Map target → binary (1 = exoplanet, 0 = not)
  3. Drop identifier / leakage columns
  4. Drop remaining object columns
  5. Drop entirely-NaN numeric columns
  6. Median imputation
  7. StandardScaler normalisation  (essential for Logistic Regression)
  8. Stratified 80/20 train-test split
"""
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


# ──────────────────────────────────────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame, cfg: dict, save_artifacts: bool = True):
    """
    Parameters
    ----------
    df  : raw DataFrame from data_loader
    cfg : plain Python dict from config.py  (keys in UPPER_CASE)
    save_artifacts : dump scaler / imputer / column list to models/

    Returns
    -------
    X, y, transformers_dict   or   None, None, {}  on failure
    """
    if df is None:
        logger.error("Input DataFrame is None.")
        return None, None, {}

    logger.info("─── Preprocessing start ───")
    df_proc = df.copy()
    df_proc = clean_column_names(df_proc)

    target_col      = cfg["TARGET_COLUMN"]
    positive_labels = cfg["POSITIVE_LABELS"]
    negative_label  = cfg["NEGATIVE_LABEL"]

    if target_col not in df_proc.columns:
        logger.error(f"Target column '{target_col}' not found in data.")
        return None, None, {}

    # ── 1. Encode target ──────────────────────────────────────────────────────
    def _encode(val):
        if val in positive_labels:
            return 1
        if val == negative_label:
            return 0
        return -1                   # unknown → will be filtered out

    df_proc["_target"] = df_proc[target_col].apply(_encode)
    df_proc = df_proc[df_proc["_target"] != -1].copy()
    if df_proc.empty:
        logger.error("No rows remain after target encoding. Check POSITIVE_LABELS / NEGATIVE_LABEL.")
        return None, None, {}

    y = df_proc["_target"].reset_index(drop=True)
    X = df_proc.drop(columns=["_target", target_col], errors="ignore")

    logger.info(f"Target distribution:\n{y.value_counts().to_string()}")

    # ── 2. Drop configured columns ────────────────────────────────────────────
    drop_cols = [c for c in cfg["FEATURES_TO_DROP"] if c in X.columns]
    X = X.drop(columns=drop_cols, errors="ignore")
    logger.info(f"Dropped {len(drop_cols)} configured columns.")

    # ── 3. Drop remaining object/string columns ───────────────────────────────
    obj_cols = X.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        logger.warning(f"Dropping {len(obj_cols)} remaining categorical columns: {obj_cols}")
        X = X.drop(columns=obj_cols)

    # ── 4. Keep only numeric columns ──────────────────────────────────────────
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    # ── 5. Drop entirely-NaN numeric columns ──────────────────────────────────
    all_nan = [c for c in num_cols if X[c].isnull().all()]
    if all_nan:
        logger.warning(f"Dropping {len(all_nan)} all-NaN columns: {all_nan}")
        X = X.drop(columns=all_nan)
        num_cols = [c for c in num_cols if c not in all_nan]

    if not num_cols:
        logger.error("No numeric features remain after cleaning.")
        return None, None, {}

    X = X[num_cols].reset_index(drop=True)

    logger.info(f"Features remaining: {len(num_cols)}")
    logger.info(f"Missing values total: {X.isnull().sum().sum():,}")

    transformers = {}

    # ── 6. Median imputation ──────────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)
    X       = pd.DataFrame(X_imp, columns=num_cols)
    transformers["imputer"] = imputer
    if save_artifacts:
        joblib.dump(imputer, cfg["IMPUTER_PATH"])
        logger.info(f"Imputer saved → {cfg['IMPUTER_PATH']}")

    # ── 7. Standard scaling ───────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    X       = pd.DataFrame(X_sc, columns=num_cols)
    transformers["scaler"] = scaler
    if save_artifacts:
        joblib.dump(scaler,          cfg["SCALER_PATH"])
        joblib.dump(num_cols,        cfg["TRAINING_COLUMNS_PATH"])
        logger.info(f"Scaler saved              → {cfg['SCALER_PATH']}")
        logger.info(f"Training columns saved    → {cfg['TRAINING_COLUMNS_PATH']}")

    logger.info(f"─── Preprocessing complete  X={X.shape}  y={y.shape} ───")
    return X, y, transformers


# ──────────────────────────────────────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """Stratified train/test split."""
    if X is None or X.empty or y is None or y.empty:
        logger.error("Cannot split: X or y is None/empty.")
        return None, None, None, None

    # Safety: drop any rows that still have NaNs
    nan_mask = X.isnull().any(axis=1)
    if nan_mask.any():
        logger.warning(f"Dropping {nan_mask.sum()} NaN rows before split.")
        X = X[~nan_mask].reset_index(drop=True)
        y = y[~nan_mask].reset_index(drop=True)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = cfg["TEST_SIZE"],
            random_state = cfg["RANDOM_STATE"],
            stratify     = y,
        )
        logger.info(
            f"Split → Train: {X_train.shape}  Test: {X_test.shape}  "
            f"| Train pos rate: {y_train.mean():.3f}  Test pos rate: {y_test.mean():.3f}"
        )
        return X_train, X_test, y_train, y_test
    except Exception as exc:
        logger.error(f"Train/test split failed: {exc}", exc_info=True)
        return None, None, None, None
