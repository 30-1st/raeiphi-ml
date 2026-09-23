"""
RAEIPHI — TensorFlow Leak Detection Models
=============================================
Two TensorFlow models for water/leak detection from CSI data:

  1. Autoencoder — trained on "dry" data only. Reconstruction error
     = anomaly score. High error = environmental change (water).

  2. Binary classifier — trained on dry vs wet labels. Direct
     classification with confidence score.

Also trains a scikit-learn baseline (Random Forest) for comparison.

Input:  ml/data/leak_train.csv, leak_val.csv, leak_test.csv
        ml/data/leak_feature_config.json
Output: ml/models/leak_autoencoder.keras
        ml/models/leak_classifier.keras
        ml/models/leak_results.json
        ml/notebooks/figures/leak_* (evaluation plots)
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score,
    roc_curve,
)

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "notebooks", "figures")

TRAIN_FILE = os.path.join(DATA_DIR, "leak_train.csv")
VAL_FILE = os.path.join(DATA_DIR, "leak_val.csv")
TEST_FILE = os.path.join(DATA_DIR, "leak_test.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "leak_feature_config.json")
RESULTS_FILE = os.path.join(MODEL_DIR, "leak_results.json")

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="darkgrid")
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LABEL_COL = "water_present"
META_COLS = ["scenario", "water_present", "water_severity", "n_people"]


def load_data():
    """Load splits and config."""
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    feature_cols = config["feature_columns"]

    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train[feature_cols].values.astype(np.float32)
    y_train = train[LABEL_COL].values
    X_val = val[feature_cols].values.astype(np.float32)
    y_val = val[LABEL_COL].values
    X_test = test[feature_cols].values.astype(np.float32)
    y_test = test[LABEL_COL].values

    # Scenario labels for analysis
    scenarios_test = test["scenario"].values if "scenario" in test.columns else None

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Features: {len(feature_cols)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scenarios_test, config


# ── Model 1: Autoencoder ───────────────────────────────────────

def build_autoencoder(n_features: int) -> Model:
    """
    Autoencoder that compresses CSI features into a bottleneck
    and reconstructs them. Trained on DRY data only.
    High reconstruction error = environmental anomaly (water).
    """
    # Encoder
    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    bottleneck = layers.Dense(32, activation="relu", name="bottleneck")(x)

    # Decoder
    x = layers.Dense(64, activation="relu")(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(n_features, activation="linear")(x)

    model = Model(inputs, outputs, name="leak_autoencoder")
    return model


def train_autoencoder(X_train, y_train, X_val, y_val, n_features):
    """Train autoencoder on DRY samples only."""
    print("\n" + "=" * 60)
    print("MODEL 1: TensorFlow Autoencoder")
    print("=" * 60)

    # Filter to dry samples only
    X_train_dry = X_train[y_train == 0]
    X_val_dry = X_val[y_val == 0]
    print(f"  Training on DRY data only: {len(X_train_dry)} samples")
    print(f"  Validation (dry): {len(X_val_dry)} samples")

    model = build_autoencoder(n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7
        ),
    ]

    start = time.time()
    history = model.fit(
        X_train_dry, X_train_dry,  # Input = output (reconstruction)
        validation_data=(X_val_dry, X_val_dry),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - start

    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Epochs: {len(history.history['loss'])}")

    return model, history, train_time


def evaluate_autoencoder(model, X_val, y_val, X_test, y_test):
    """
    Evaluate autoencoder by using reconstruction error as anomaly score.
    Higher error = more likely to be water/leak.
    """
    print("\n  Evaluating autoencoder...")

    # Compute reconstruction error
    val_reconstructed = model.predict(X_val, verbose=0)
    val_errors = np.mean(np.square(X_val - val_reconstructed), axis=1)

    test_reconstructed = model.predict(X_test, verbose=0)
    test_errors = np.mean(np.square(X_test - test_reconstructed), axis=1)

    # Set threshold using validation data
    # Use the 95th percentile of DRY reconstruction errors
    dry_val_errors = val_errors[y_val == 0]
    threshold = np.percentile(dry_val_errors, 95)
    print(f"  Threshold (95th pct of dry errors): {threshold:.6f}")

    # Predict on test set
    test_predictions = (test_errors > threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(y_test, test_predictions, zero_division=0)
    recall = recall_score(y_test, test_predictions, zero_division=0)
    f1 = f1_score(y_test, test_predictions, zero_division=0)
    auc = roc_auc_score(y_test, test_errors)

    print(f"\n  Autoencoder Test Results:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
        "threshold": float(threshold),
    }

    return metrics, test_errors, test_predictions, threshold


# ── Model 2: Binary Classifier ─────────────────────────────────

def build_classifier(n_features: int) -> Model:
    """
    Binary classifier: dry vs wet.
    Direct classification with sigmoid output.
    """
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="leak_classifier")

    return model


def train_classifier(X_train, y_train, X_val, y_val, n_features):
    """Train binary classifier on all labeled data."""
    print("\n" + "=" * 60)
    print("MODEL 2: TensorFlow Binary Classifier")
    print("=" * 60)

    # Class weights for any imbalance
    n_dry = np.sum(y_train == 0)
    n_wet = np.sum(y_train == 1)
    weight_dry = len(y_train) / (2 * n_dry)
    weight_wet = len(y_train) / (2 * n_wet)
    class_weights = {0: weight_dry, 1: weight_wet}
    print(f"  Class weights: dry={weight_dry:.2f}, wet={weight_wet:.2f}")

    model = build_classifier(n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7
        ),
    ]

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    train_time = time.time() - start

    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Epochs: {len(history.history['loss'])}")

    return model, history, train_time


def evaluate_classifier(model, X_test, y_test):
    """Evaluate the binary classifier."""
    print("\n  Evaluating classifier...")

    probabilities = model.predict(X_test, verbose=0).flatten()
    predictions = (probabilities > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    print(f"\n  Classifier Test Results:")
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, predictions,
                                target_names=["dry", "wet"]))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
    }

    return metrics, probabilities, predictions


# ── Baseline: Scikit-learn ──────────────────────────────────────

def train_baseline(X_train, y_train, X_test, y_test):
    """Train a Random Forest baseline for comparison."""
    print("\n" + "=" * 60)
    print("BASELINE: Scikit-learn Random Forest")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    start = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start

    predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    print(f"  Training time: {train_time:.1f}s")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc),
        "train_time": train_time,
    }

    return metrics, probabilities, predictions, rf


# ── Visualizations ──────────────────────────────────────────────

def plot_autoencoder_errors(test_errors, y_test, threshold):
    """Distribution of reconstruction errors for dry vs wet."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(test_errors[y_test == 0], bins=50, alpha=0.6, label="Dry", color="#4CAF50")
    ax.hist(test_errors[y_test == 1], bins=50, alpha=0.6, label="Wet", color="#F44336")
    ax.axvline(x=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder Reconstruction Error — Dry vs Wet")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "leak_01_autoencoder_errors.png"), dpi=150)
    plt.close()
    print(f"  Saved: leak_01_autoencoder_errors.png")


def plot_roc_curves(results_dict):
    """ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"Autoencoder": "#FF9800", "Classifier": "#4CAF50", "Random Forest": "#2196F3"}

    for name, data in results_dict.items():
        if "fpr" in data and "tpr" in data:
            ax.plot(data["fpr"], data["tpr"],
                    label=f"{name} (AUC={data['auc']:.3f})",
                    color=colors.get(name, "gray"), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Leak Detection Models")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "leak_02_roc_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved: leak_02_roc_curves.png")


def plot_confusion_matrices(results_dict, y_test):
    """Side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = ["Autoencoder", "Classifier", "Random Forest"]

    for ax, name in zip(axes, names):
        if name in results_dict and "predictions" in results_dict[name]:
            cm = confusion_matrix(y_test, results_dict[name]["predictions"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Dry", "Wet"], yticklabels=["Dry", "Wet"])
            ax.set_title(f"{name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — Leak Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "leak_03_confusion_matrices.png"), dpi=150)
    plt.close()
    print(f"  Saved: leak_03_confusion_matrices.png")


def plot_model_comparison(all_metrics):
    """Bar chart comparing all models."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    colors = ["#FF9800", "#4CAF50", "#2196F3"]
    model_names = ["Autoencoder", "Classifier", "Random Forest"]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        if name in all_metrics:
            vals = [all_metrics[name].get(m, 0) for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Leak Detection — Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "leak_04_model_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved: leak_04_model_comparison.png")


# ── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RAEIPHI — TensorFlow Leak Detection")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")

    X_train, y_train, X_val, y_val, X_test, y_test, scenarios, config = load_data()
    n_features = config["n_features"]

    all_metrics = {}
    roc_data = {}

    # ── Autoencoder ─────────────────────────────────────────────
    ae_model, ae_history, ae_time = train_autoencoder(
        X_train, y_train, X_val, y_val, n_features
    )
    ae_metrics, ae_errors, ae_preds, ae_threshold = evaluate_autoencoder(
        ae_model, X_val, y_val, X_test, y_test
    )
    ae_metrics["train_time"] = ae_time
    all_metrics["Autoencoder"] = ae_metrics

    fpr, tpr, _ = roc_curve(y_test, ae_errors)
    roc_data["Autoencoder"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                                "auc": ae_metrics["auc_roc"], "predictions": ae_preds}

    # Save autoencoder
    ae_path = os.path.join(MODEL_DIR, "leak_autoencoder.keras")
    ae_model.save(ae_path)
    print(f"\n  Saved: {ae_path}")

    # ── Classifier ──────────────────────────────────────────────
    clf_model, clf_history, clf_time = train_classifier(
        X_train, y_train, X_val, y_val, n_features
    )
    clf_metrics, clf_probs, clf_preds = evaluate_classifier(
        clf_model, X_test, y_test
    )
    clf_metrics["train_time"] = clf_time
    all_metrics["Classifier"] = clf_metrics

    fpr, tpr, _ = roc_curve(y_test, clf_probs)
    roc_data["Classifier"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                               "auc": clf_metrics["auc_roc"], "predictions": clf_preds}

    # Save classifier
    clf_path = os.path.join(MODEL_DIR, "leak_classifier.keras")
    clf_model.save(clf_path)
    print(f"\n  Saved: {clf_path}")

    # ── Baseline ────────────────────────────────────────────────
    rf_metrics, rf_probs, rf_preds, rf_model = train_baseline(
        X_train, y_train, X_test, y_test
    )
    all_metrics["Random Forest"] = rf_metrics

    fpr, tpr, _ = roc_curve(y_test, rf_probs)
    roc_data["Random Forest"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                                  "auc": rf_metrics["auc_roc"], "predictions": rf_preds}

    # ── Visualizations ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)
    plot_autoencoder_errors(ae_errors, y_test, ae_threshold)
    plot_roc_curves(roc_data)
    plot_confusion_matrices(roc_data, y_test)
    plot_model_comparison(all_metrics)

    # ── Save results ────────────────────────────────────────────
    results = {
        "Autoencoder": {**ae_metrics, "architecture": "Dense(128→64→32→64→128→n)",
                        "threshold": float(ae_threshold), "framework": "tensorflow"},
        "Classifier": {**clf_metrics, "architecture": "Dense(128→64→32→1, sigmoid)",
                       "framework": "tensorflow"},
        "Random Forest": {**rf_metrics, "framework": "scikit-learn"},
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("LEAK DETECTION — FINAL COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print(f"  {'─' * 58}")
    for name in ["Autoencoder", "Classifier", "Random Forest"]:
        m = all_metrics[name]
        print(f"  {name:<18} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc_roc']:>10.4f}")

    best = max(all_metrics.items(), key=lambda x: x[1]["f1"])
    print(f"\n  Best model: {best[0]} (F1: {best[1]['f1']:.4f})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
