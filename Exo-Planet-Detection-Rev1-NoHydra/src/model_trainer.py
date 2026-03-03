"""
model_trainer.py  –  Revision 1: Logistic Regression
Plots produced (saved to reports/):
  • confusion_matrix.png
  • roc_curve.png
  • pr_curve.png
  • metrics_summary.png
  • feature_coefficients.png
"""
import logging
import os
import re

import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – works on all platforms
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ── colour palette ─────────────────────────────────────────────────────────────
_C = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "purple": "#9467bd",
    "brown":  "#8c564b",
    "pink":   "#e377c2",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_name(s: str) -> str:
    s = re.sub(r"[()\s/]", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_.-]", "", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_")


def _save_fig(fig, reports_dir: str, filename: str) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, filename)
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {path}")
    except Exception as exc:
        logger.error(f"Could not save {filename}: {exc}")
    finally:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────
def get_model_instance(cfg: dict) -> LogisticRegression:
    name   = cfg["MODEL_NAME"]
    params = dict(cfg["MODEL_PARAMS"])
    if name != "LogisticRegression":
        raise ValueError(f"This trainer only supports LogisticRegression, got '{name}'.")
    return LogisticRegression(**params)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict):
    """Fit model, save to disk, return fitted model."""
    if X_train is None or X_train.empty:
        logger.error("Training data is empty. Aborting.")
        return None

    logger.info(f"Training {cfg['MODEL_NAME']}  |  n_samples={len(X_train):,}  |  n_features={X_train.shape[1]}")
    model = get_model_instance(cfg)
    try:
        model.fit(X_train, y_train)
        logger.info("Training complete.")
    except Exception as exc:
        logger.error(f"Training error: {exc}", exc_info=True)
        return None

    try:
        os.makedirs(cfg["MODEL_DIR"], exist_ok=True)
        joblib.dump(model, cfg["MODEL_PATH"])
        logger.info(f"Model saved → {cfg['MODEL_PATH']}")
    except Exception as exc:
        logger.error(f"Could not save model: {exc}", exc_info=True)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, cfg: dict) -> dict:
    """
    Compute all metrics, print classification report, save all plots.
    Returns a flat dict of metric names → float values.
    """
    if model is None or X_test is None or X_test.empty:
        logger.error("Model or test data missing. Skipping evaluation.")
        return {}

    logger.info("─── Evaluation start ───")
    metrics: dict = {}

    y_pred       = model.predict(X_test)
    y_proba      = model.predict_proba(X_test)[:, 1]

    # ── Classification report ──────────────────────────────────────────────
    report_str  = classification_report(
        y_test, y_pred,
        target_names=["Not Exoplanet (0)", "Exoplanet (1)"],
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Not Exoplanet (0)", "Exoplanet (1)"],
        output_dict=True,
    )
    logger.info(f"\n{report_str}")

    # Flatten report into metrics dict
    for label, scores in report_dict.items():
        if isinstance(scores, dict):
            for metric_name, val in scores.items():
                metrics[_safe_name(f"{label}_{metric_name}")] = val
        else:
            metrics[_safe_name(str(label))] = scores

    # ── Scalar metrics ─────────────────────────────────────────────────────
    metrics["accuracy"] = accuracy_score(y_test, y_pred)
    metrics["roc_auc"]  = roc_auc_score(y_test, y_proba)

    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    metrics["pr_auc"] = auc(rec_vals, prec_vals)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["precision"]   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["f1_exoplanet"] = report_dict.get("Exoplanet (1)", {}).get("f1-score", 0.0)

    # Print summary table
    logger.info("─── Performance summary ─────────────────────────────────────")
    for k in ["accuracy", "sensitivity", "specificity", "precision", "f1_exoplanet", "roc_auc", "pr_auc"]:
        logger.info(f"  {k:<20s}: {metrics[k]:.4f}")
    logger.info("─────────────────────────────────────────────────────────────")

    reports_dir = cfg["REPORTS_DIR"]

    # ── Generate & save all plots ──────────────────────────────────────────
    _plot_confusion_matrix(cm, metrics, reports_dir)
    _plot_roc_curve(y_test, y_proba, metrics["roc_auc"], reports_dir)
    _plot_pr_curve(prec_vals, rec_vals, metrics["pr_auc"], reports_dir)
    _plot_metrics_summary(metrics, reports_dir)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Feature coefficients
# ─────────────────────────────────────────────────────────────────────────────
def get_feature_importances(model, feature_names: list, cfg: dict, top_n: int = 25):
    if not hasattr(model, "coef_"):
        logger.warning("Model has no coef_ attribute — skipping feature importance plot.")
        return None

    coefs = model.coef_[0]
    df    = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_coef"] = df["coefficient"].abs()
    df    = df.nlargest(top_n, "abs_coef").reset_index(drop=True)

    logger.info(f"\nTop {top_n} feature coefficients:\n{df[['feature','coefficient']].to_string(index=False)}")

    colours = [_C["blue"] if c > 0 else _C["orange"] for c in df["coefficient"]]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.40)))
    ax.barh(df["feature"], df["coefficient"], color=colours, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Coefficients  –  Logistic Regression",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    patches = [
        mpatches.Patch(color=_C["blue"],   label="Positive effect (→ Exoplanet)"),
        mpatches.Patch(color=_C["orange"], label="Negative effect (→ Not Exoplanet)"),
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    _save_fig(fig, cfg["REPORTS_DIR"], "feature_coefficients.png")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Private plot functions
# ─────────────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(cm: np.ndarray, metrics: dict, reports_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Exoplanet", "Exoplanet"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    sens = metrics.get("sensitivity", 0)
    spec = metrics.get("specificity", 0)
    ax.set_title(
        f"Confusion Matrix  –  Logistic Regression\n"
        f"Sensitivity (Recall) = {sens:.4f}   |   Specificity = {spec:.4f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, reports_dir, "confusion_matrix.png")


def _plot_roc_curve(y_test, y_proba, roc_auc_val: float, reports_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=_C["blue"], lw=2,
            label=f"Logistic Regression  (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.10, color=_C["blue"])
    ax.set_xlabel("False Positive Rate",            fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curve  –  Logistic Regression", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.30)
    plt.tight_layout()
    _save_fig(fig, reports_dir, "roc_curve.png")


def _plot_pr_curve(precision: np.ndarray, recall: np.ndarray,
                   pr_auc_val: float, reports_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color=_C["orange"], lw=2,
            label=f"Logistic Regression  (AUC = {pr_auc_val:.4f})")
    ax.fill_between(recall, precision, alpha=0.10, color=_C["orange"])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision",            fontsize=11)
    ax.set_title("Precision-Recall Curve  –  Logistic Regression",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.30)
    plt.tight_layout()
    _save_fig(fig, reports_dir, "pr_curve.png")


def _plot_metrics_summary(metrics: dict, reports_dir: str) -> None:
    keys   = ["accuracy", "sensitivity", "specificity", "precision", "f1_exoplanet", "roc_auc", "pr_auc"]
    labels = ["Accuracy", "Sensitivity\n(Recall)", "Specificity", "Precision",
              "F1\n(Exoplanet)", "ROC-AUC", "PR-AUC"]
    values  = [metrics.get(k, 0.0) for k in keys]
    colours = [_C["blue"], _C["green"], _C["orange"], _C["purple"],
               _C["brown"], _C["pink"], "#17becf"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.6, width=0.60)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Summary  –  Logistic Regression",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.30)
    plt.tight_layout()
    _save_fig(fig, reports_dir, "metrics_summary.png")
