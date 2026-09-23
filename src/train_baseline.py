"""
RAEIPHI — Scikit-learn Baseline Models
========================================
Trains Random Forest and Gradient Boosting classifiers on the
processed CSI dataset. These establish the performance baseline
that the PyTorch model in Phase 1.4 needs to beat.

Input:  ml/data/train.csv, ml/data/val.csv, ml/data/feature_config.json
Output: ml/models/random_forest.pkl, ml/models/gradient_boosting.pkl
        ml/models/baseline_results.json
        ml/notebooks/figures/08–12 (evaluation plots)
"""

import pandas as pd
import numpy as np
import json
import os
import time
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Config ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "figures")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VAL_FILE = os.path.join(DATA_DIR, "val.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "feature_config.json")
RESULTS_FILE = os.path.join(MODEL_DIR, "baseline_results.json")

RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")


def load_data():
    """Load training and validation sets + feature config."""
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    feature_cols = config["feature_columns"]

    X_train = train_df[feature_cols].values
    y_train = train_df["occupancy"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["occupancy"].values

    print(f"Training set:   {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
    print(f"Classes:        {config['classes']}")

    return X_train, y_train, X_val, y_val, config


def evaluate_model(model, X_val, y_val, model_name: str) -> dict:
    """Run full evaluation on validation set and return metrics."""
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_val, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─' * 50}")
    print(f"  {model_name} — Validation Results")
    print(f"{'─' * 50}")
    print(f"  Accuracy:           {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"  Precision (macro):  {precision_macro:.4f}")
    print(f"  Recall (macro):     {recall_macro:.4f}")
    print(f"  F1 Score (macro):   {f1_macro:.4f}")
    print(f"  F1 Score (weighted):{f1_weighted:.4f}")

    print(f"\n  Per-class report:")
    report = classification_report(y_val, y_pred, zero_division=0)
    for line in report.split("\n"):
        print(f"  {line}")

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }

    return metrics, y_pred


def plot_confusion_matrix(y_val, y_pred, classes, model_name: str, filename: str):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted Occupancy", fontsize=12)
    ax.set_ylabel("Actual Occupancy", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_f1(results: dict, classes: list):
    """Bar chart comparing per-class F1 scores between models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(classes))
    width = 0.35

    rf_f1s = results["Random Forest"]["per_class_f1"]
    gb_f1s = results["Gradient Boosting"]["per_class_f1"]

    bars1 = ax.bar(x - width / 2, rf_f1s, width, label="Random Forest", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, gb_f1s, width, label="Gradient Boosting", color="#FF9800", alpha=0.85)

    ax.set_xlabel("Occupancy Count", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score — Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4, label="90% target")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "10_per_class_f1_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_cols, model_name: str, filename: str, top_n: int = 25):
    """Horizontal bar chart of top feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="#4CAF50", alpha=0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_accuracy_comparison(results: dict):
    """Side-by-side bar chart of overall metrics for both models."""
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "F1\n(weighted)"]

    rf_vals = [results["Random Forest"]["metrics"][m] for m in metrics]
    gb_vals = [results["Gradient Boosting"]["metrics"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, rf_vals, width, label="Random Forest", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, gb_vals, width, label="Gradient Boosting", color="#FF9800", alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Overall Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "12_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("RAEIPHI — Scikit-learn Baseline Training")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────
    X_train, y_train, X_val, y_val, config = load_data()
    classes = config["classes"]
    feature_cols = config["feature_columns"]

    results = {}

    # ── Model 1: Random Forest ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Training Random Forest...")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,          # Let trees grow fully
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",     # sqrt(n_features) per split
        class_weight="balanced", # Handle class imbalance
        random_state=RANDOM_SEED,
        n_jobs=-1,               # Use all CPU cores
    )

    start = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - start
    print(f"  Training time: {rf_train_time:.1f}s")

    rf_metrics, rf_pred = evaluate_model(rf, X_val, y_val, "Random Forest")

    # Per-class F1
    rf_per_class_f1 = f1_score(y_val, rf_pred, labels=classes, average=None, zero_division=0).tolist()

    results["Random Forest"] = {
        "metrics": rf_metrics,
        "per_class_f1": rf_per_class_f1,
        "train_time_seconds": rf_train_time,
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
        },
    }

    # ── Model 2: Gradient Boosting ──────────────────────────────
    print("\n" + "=" * 60)
    print("Training Gradient Boosting...")
    print("=" * 60)

    gb = HistGradientBoostingClassifier(
        max_iter=300,            # Number of boosting iterations
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=5,
        max_bins=255,            # Histogram bins for speed
        class_weight="balanced", # Handle class imbalance
        random_state=RANDOM_SEED,
    )

    start = time.time()
    gb.fit(X_train, y_train)
    gb_train_time = time.time() - start
    print(f"  Training time: {gb_train_time:.1f}s")

    gb_metrics, gb_pred = evaluate_model(gb, X_val, y_val, "Gradient Boosting")

    # Per-class F1
    gb_per_class_f1 = f1_score(y_val, gb_pred, labels=classes, average=None, zero_division=0).tolist()

    results["Gradient Boosting"] = {
        "metrics": gb_metrics,
        "per_class_f1": gb_per_class_f1,
        "train_time_seconds": gb_train_time,
        "hyperparameters": {
            "max_iter": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_samples_leaf": 5,
            "max_bins": 255,
            "class_weight": "balanced",
        },
    }

    # ── Save models ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving models and generating visualizations...")
    print("=" * 60)

    rf_path = os.path.join(MODEL_DIR, "random_forest.pkl")
    gb_path = os.path.join(MODEL_DIR, "gradient_boosting.pkl")

    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    rf_size = os.path.getsize(rf_path) / 1024 / 1024
    print(f"  Random Forest saved: {rf_path} ({rf_size:.1f} MB)")

    with open(gb_path, "wb") as f:
        pickle.dump(gb, f)
    gb_size = os.path.getsize(gb_path) / 1024 / 1024
    print(f"  Gradient Boosting saved: {gb_path} ({gb_size:.1f} MB)")

    # ── Generate visualizations ─────────────────────────────────
    print()
    plot_confusion_matrix(y_val, rf_pred, classes, "Random Forest",
                          "08_confusion_matrix_random_forest.png")
    plot_confusion_matrix(y_val, gb_pred, classes, "Gradient Boosting",
                          "09_confusion_matrix_gradient_boosting.png")
    plot_per_class_f1(results, classes)
    plot_feature_importance(rf, feature_cols, "Random Forest",
                            "11a_feature_importance_rf.png")
    # HistGradientBoosting doesn't support feature_importances_ without
    # additional config, so we skip its importance plot. The RF importance
    # plot covers feature ranking. For GB feature importance, use
    # permutation_importance (slower but model-agnostic).
    # plot_feature_importance(gb, feature_cols, "Gradient Boosting",
    #                         "11b_feature_importance_gb.png")
    plot_accuracy_comparison(results)

    # ── Save results JSON ───────────────────────────────────────
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {RESULTS_FILE}")

    # ── Final comparison ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<22} {'Random Forest':>15} {'Gradient Boost':>15}")
    print(f"  {'─' * 52}")
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        rf_val = results["Random Forest"]["metrics"][metric]
        gb_val = results["Gradient Boosting"]["metrics"][metric]
        winner = " ◀" if rf_val > gb_val else ""
        winner_gb = " ◀" if gb_val > rf_val else ""
        label = metric.replace("_", " ").title()
        print(f"  {label:<22} {rf_val:>13.4f}{winner}  {gb_val:>13.4f}{winner_gb}")

    print(f"\n  {'Training time':<22} {rf_train_time:>13.1f}s  {gb_train_time:>13.1f}s")

    # Identify best model
    rf_f1 = results["Random Forest"]["metrics"]["f1_weighted"]
    gb_f1 = results["Gradient Boosting"]["metrics"]["f1_weighted"]
    best = "Random Forest" if rf_f1 >= gb_f1 else "Gradient Boosting"
    best_f1 = max(rf_f1, gb_f1)

    print(f"\n  Best baseline: {best} (F1 weighted: {best_f1:.4f})")
    print(f"  PyTorch model target: beat {best_f1:.4f} F1 weighted")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
