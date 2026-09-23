"""
RAEIPHI — PyTorch Occupancy Classification Model
==================================================
Trains a deep neural network on the CSI dataset to predict
room occupancy count (0–10). Two architectures are trained
and compared:

  1. Fully Connected Network (FCN) — baseline deep learning
  2. 1D CNN — treats subcarrier features as a spatial signal

Both are evaluated against the scikit-learn baselines from
Phase 1.3. The best model is saved for deployment.

Input:  ml/data/train.csv, ml/data/val.csv, ml/data/test.csv
        ml/data/feature_config.json
        ml/models/baseline_results.json
Output: ml/models/best_pytorch_model.pt
        ml/models/pytorch_results.json
        ml/notebooks/figures/13–17 (training curves, evaluation)
"""

import pandas as pd
import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
CONFIG_FILE = os.path.join(DATA_DIR, "feature_config.json")
BASELINE_FILE = os.path.join(MODEL_DIR, "baseline_results.json")
RESULTS_FILE = os.path.join(MODEL_DIR, "pytorch_results.json")

RANDOM_SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 15     # Stop if val loss doesn't improve for N epochs
WEIGHT_DECAY = 1e-4           # L2 regularization

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="darkgrid")

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ─────────────────────────────────────────────────────
class CSIDataset(Dataset):
    """PyTorch Dataset for CSI occupancy data."""

    def __init__(self, csv_path: str, feature_cols: list):
        df = pd.read_csv(csv_path)
        self.features = torch.FloatTensor(df[feature_cols].values.copy())
        self.labels = torch.LongTensor(df["occupancy"].values.copy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ── Model Architectures ────────────────────────────────────────
class OccupancyFCN(nn.Module):
    """
    Fully Connected Network for occupancy classification.
    Architecture: Input → 256 → 128 → 64 → 32 → 11 classes
    Uses BatchNorm, ReLU, and Dropout for regularization.
    """

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.network(x)


class OccupancyCNN(nn.Module):
    """
    1D Convolutional Network for occupancy classification.
    Treats the feature vector as a 1D signal — groups of
    subcarrier features form spatial patterns the CNN can detect.

    Architecture:
      Input (1, n_features)
      → Conv1d(32, kernel=5) → BN → ReLU → Pool
      → Conv1d(64, kernel=3) → BN → ReLU → Pool
      → Conv1d(128, kernel=3) → BN → ReLU → AdaptivePool
      → FC(128) → FC(64) → FC(11)
    """

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Conv block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Conv block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # Fixed output size regardless of input
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # Reshape for Conv1d: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ── Training Loop ──────────────────────────────────────────────
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    One full pass through the training data.
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set without gradient computation.
    Returns average loss, accuracy, and all predictions.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, model_name, n_epochs, lr, device):
    """
    Full training loop with early stopping and learning rate scheduling.
    Returns training history and the best model state.
    """
    # Class weights for imbalanced data
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    train_labels = np.array(train_labels)

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n{'─' * 60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print(f"  Epochs: {n_epochs} (early stop patience: {EARLY_STOP_PATIENCE})")
    print(f"{'─' * 60}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Train Acc':>10}  {'Val Acc':>10}  {'LR':>10}")
    print(f"  {'─' * 58}")

    start_time = time.time()

    for epoch in range(n_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " ★"
        else:
            patience_counter += 1
            marker = ""

        # Print progress every 10 epochs or on improvement
        if (epoch + 1) % 10 == 0 or marker or epoch == 0:
            print(f"  {epoch + 1:>5}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
                  f"{train_acc:>9.4f}  {val_acc:>9.4f}  {current_lr:>10.6f}{marker}")

        # Stop if no improvement
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch + 1} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    train_time = time.time() - start_time
    print(f"\n  Training time: {train_time:.1f}s ({epoch + 1} epochs)")

    # Restore best model
    model.load_state_dict(best_model_state)
    history["train_time"] = train_time
    history["best_epoch"] = int(np.argmin(history["val_loss"]) + 1)
    history["total_epochs"] = epoch + 1

    return model, history


# ── Evaluation ──────────────────────────────────────────────────
def full_evaluation(model, dataloader, device, model_name, classes):
    """Run complete evaluation and return metrics dict."""
    criterion = nn.CrossEntropyLoss()
    _, accuracy, y_pred, y_true = validate(model, dataloader, criterion, device)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─' * 50}")
    print(f"  {model_name} — Validation Results")
    print(f"{'─' * 50}")
    print(f"  Accuracy:           {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"  Precision (macro):  {precision_macro:.4f}")
    print(f"  Recall (macro):     {recall_macro:.4f}")
    print(f"  F1 Score (macro):   {f1_macro:.4f}")
    print(f"  F1 Score (weighted):{f1_weighted:.4f}")

    print(f"\n  Per-class report:")
    report = classification_report(y_true, y_pred, zero_division=0)
    for line in report.split("\n"):
        print(f"  {line}")

    per_class_f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0).tolist()

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_f1": per_class_f1,
    }

    return metrics, y_pred, y_true


# ── Visualizations ──────────────────────────────────────────────
def plot_training_curves(histories: dict):
    """Loss and accuracy curves for both architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {"FCN": "#2196F3", "1D CNN": "#FF9800"}

    # Loss curves
    for name, hist in histories.items():
        axes[0].plot(hist["train_loss"], label=f"{name} Train", color=colors[name], alpha=0.7)
        axes[0].plot(hist["val_loss"], label=f"{name} Val", color=colors[name],
                     linestyle="--", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].set_yscale("log")

    # Accuracy curves
    for name, hist in histories.items():
        axes[1].plot(hist["train_acc"], label=f"{name} Train", color=colors[name], alpha=0.7)
        axes[1].plot(hist["val_acc"], label=f"{name} Val", color=colors[name],
                     linestyle="--", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].axhline(y=0.95, color="red", linestyle=":", alpha=0.4, label="Baseline target")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "13_training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, classes, model_name, filename):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=classes, yticklabels=classes,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Occupancy", fontsize=12)
    ax.set_ylabel("Actual Occupancy", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_all_models_comparison(baseline_results, pytorch_results, classes):
    """Bar chart comparing all 4 models on key metrics."""
    all_models = {}
    for name in ["Random Forest", "Gradient Boosting"]:
        all_models[name] = baseline_results[name]["metrics"]
    for name in ["FCN", "1D CNN"]:
        all_models[f"PyTorch {name}"] = pytorch_results[name]["metrics"]

    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "F1\n(weighted)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    x = np.arange(len(metrics))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (name, m) in enumerate(all_models.items()):
        vals = [m[metric] for metric in metrics]
        bars = ax.bar(x + offsets[i] * width, vals, width,
                      label=name, color=colors[i], alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("All Models Comparison — Scikit-learn Baselines vs PyTorch",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "16_all_models_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_all_models(baseline_results, pytorch_results, classes):
    """Per-class F1 comparison across all models."""
    all_f1s = {
        "Random Forest": baseline_results["Random Forest"]["per_class_f1"],
        "Gradient Boosting": baseline_results["Gradient Boosting"]["per_class_f1"],
        "PyTorch FCN": pytorch_results["FCN"]["per_class_f1"],
        "PyTorch 1D CNN": pytorch_results["1D CNN"]["per_class_f1"],
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(classes))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for i, (name, f1s) in enumerate(all_f1s.items()):
        ax.bar(x + offsets[i] * width, f1s, width,
               label=name, color=colors[i], alpha=0.85)

    ax.set_xlabel("Occupancy Count", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score — All Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "17_per_class_f1_all_models.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RAEIPHI — PyTorch Occupancy Classification")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── Load config and data ────────────────────────────────────
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    feature_cols = config["feature_columns"]
    n_features = config["n_features"]
    n_classes = config["n_classes"]
    classes = config["classes"]

    print(f"Features: {n_features}")
    print(f"Classes:  {n_classes} ({classes})")

    # Create datasets and dataloaders
    train_dataset = CSIDataset(TRAIN_FILE, feature_cols)
    val_dataset = CSIDataset(VAL_FILE, feature_cols)
    test_dataset = CSIDataset(TEST_FILE, feature_cols)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset):,} samples, {len(test_loader)} batches")

    # ── Train both architectures ────────────────────────────────
    pytorch_results = {}
    histories = {}

    # Model 1: Fully Connected Network
    print("\n" + "=" * 60)
    print("ARCHITECTURE 1: Fully Connected Network (FCN)")
    print("=" * 60)

    fcn = OccupancyFCN(n_features, n_classes).to(DEVICE)
    fcn, fcn_history = train_model(
        fcn, train_loader, val_loader, "FCN",
        NUM_EPOCHS, LEARNING_RATE, DEVICE
    )
    histories["FCN"] = fcn_history

    fcn_metrics, fcn_pred, fcn_true = full_evaluation(
        fcn, val_loader, DEVICE, "PyTorch FCN", classes
    )
    pytorch_results["FCN"] = {
        "metrics": fcn_metrics,
        "per_class_f1": fcn_metrics["per_class_f1"],
        "train_time": fcn_history["train_time"],
        "best_epoch": fcn_history["best_epoch"],
        "total_epochs": fcn_history["total_epochs"],
        "architecture": "FCN (256→128→64→32→11)",
        "parameters": sum(p.numel() for p in fcn.parameters()),
    }

    # Model 2: 1D CNN
    print("\n" + "=" * 60)
    print("ARCHITECTURE 2: 1D Convolutional Network (CNN)")
    print("=" * 60)

    cnn = OccupancyCNN(n_features, n_classes).to(DEVICE)
    cnn, cnn_history = train_model(
        cnn, train_loader, val_loader, "1D CNN",
        NUM_EPOCHS, LEARNING_RATE, DEVICE
    )
    histories["1D CNN"] = cnn_history

    cnn_metrics, cnn_pred, cnn_true = full_evaluation(
        cnn, val_loader, DEVICE, "PyTorch 1D CNN", classes
    )
    pytorch_results["1D CNN"] = {
        "metrics": cnn_metrics,
        "per_class_f1": cnn_metrics["per_class_f1"],
        "train_time": cnn_history["train_time"],
        "best_epoch": cnn_history["best_epoch"],
        "total_epochs": cnn_history["total_epochs"],
        "architecture": "Conv1d(32→64→128) → FC(128→64→11)",
        "parameters": sum(p.numel() for p in cnn.parameters()),
    }

    # ── Determine best PyTorch model ────────────────────────────
    fcn_f1 = fcn_metrics["f1_weighted"]
    cnn_f1 = cnn_metrics["f1_weighted"]
    best_name = "FCN" if fcn_f1 >= cnn_f1 else "1D CNN"
    best_model = fcn if best_name == "FCN" else cnn
    best_f1 = max(fcn_f1, cnn_f1)

    # ── Save best model ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving best model and generating visualizations...")
    print("=" * 60)

    model_path = os.path.join(MODEL_DIR, "best_pytorch_model.pt")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "architecture": best_name,
        "n_features": n_features,
        "n_classes": n_classes,
        "feature_columns": feature_cols,
        "classes": classes,
        "metrics": pytorch_results[best_name]["metrics"],
    }, model_path)
    model_size = os.path.getsize(model_path) / 1024
    print(f"  Best model saved: {model_path} ({model_size:.0f} KB)")

    # ── Generate visualizations ─────────────────────────────────
    print()
    plot_training_curves(histories)
    plot_confusion_matrix(fcn_true, fcn_pred, classes, "PyTorch FCN",
                          "14_confusion_matrix_fcn.png")
    plot_confusion_matrix(cnn_true, cnn_pred, classes, "PyTorch 1D CNN",
                          "15_confusion_matrix_cnn.png")

    # Compare against baselines
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            baseline_results = json.load(f)
        plot_all_models_comparison(baseline_results, pytorch_results, classes)
        plot_per_class_all_models(baseline_results, pytorch_results, classes)

        # Baseline comparison
        baseline_best_f1 = max(
            baseline_results["Random Forest"]["metrics"]["f1_weighted"],
            baseline_results["Gradient Boosting"]["metrics"]["f1_weighted"],
        )
        beat_baseline = best_f1 > baseline_best_f1
    else:
        baseline_best_f1 = None
        beat_baseline = None

    # ── Run on held-out test set ────────────────────────────────
    print(f"\n{'=' * 60}")
    print("HELD-OUT TEST SET EVALUATION (final, unbiased)")
    print(f"{'=' * 60}")
    test_metrics, test_pred, test_true = full_evaluation(
        best_model, test_loader, DEVICE, f"PyTorch {best_name} (TEST)", classes
    )

    pytorch_results["test_set"] = {
        "model": best_name,
        "metrics": test_metrics,
        "per_class_f1": test_metrics["per_class_f1"],
    }

    # ── Save all results ────────────────────────────────────────
    # Convert any non-serializable values
    for key in pytorch_results:
        if isinstance(pytorch_results[key].get("parameters"), (int, np.integer)):
            pytorch_results[key]["parameters"] = int(pytorch_results[key].get("parameters", 0))

    with open(RESULTS_FILE, "w") as f:
        json.dump(pytorch_results, f, indent=2, default=str)
    print(f"\n  Results saved: {RESULTS_FILE}")

    # ── Final Summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PYTORCH TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<20} {'F1 Weighted':>12} {'Accuracy':>10} {'Params':>10} {'Time':>8}")
    print(f"  {'─' * 62}")

    for name in ["FCN", "1D CNN"]:
        r = pytorch_results[name]
        print(f"  PyTorch {name:<11} {r['metrics']['f1_weighted']:>12.4f} "
              f"{r['metrics']['accuracy']:>10.4f} "
              f"{r['parameters']:>10,} "
              f"{r['train_time']:>7.1f}s")

    if baseline_best_f1:
        print(f"\n  Baseline best (Gradient Boosting): {baseline_best_f1:.4f} F1 weighted")
        print(f"  PyTorch best ({best_name}):          {best_f1:.4f} F1 weighted")
        if beat_baseline:
            improvement = (best_f1 - baseline_best_f1) / baseline_best_f1 * 100
            print(f"  ✓ PyTorch BEATS baseline by {improvement:.2f}%")
        else:
            gap = (baseline_best_f1 - best_f1) / baseline_best_f1 * 100
            print(f"  ✗ PyTorch trails baseline by {gap:.2f}% — consider tuning hyperparameters")

    print(f"\n  Test set F1 weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
