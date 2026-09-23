"""
RAEIPHI — MLflow Experiment Tracking
======================================
Retroactively logs all experiments from Phases 1.3 and 1.4
into MLflow. Tracks hyperparameters, evaluation metrics,
training curves, confusion matrices, and registers the
best model in the MLflow Model Registry.

Also provides a reusable tracking function for future
training runs so new experiments are logged automatically.

Prerequisites:
  pip install mlflow

Usage:
  python src/track_experiments.py

Then view the dashboard:
  mlflow ui --port 5000

Open http://localhost:5000 in your browser.
"""

import json
import os
import pickle
import sys

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np

# ── Paths ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "notebooks", "figures")
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_ARTIFACTS = os.path.join(BASE_DIR, "mlruns", "artifacts")

BASELINE_RESULTS = os.path.join(MODEL_DIR, "baseline_results.json")
PYTORCH_RESULTS = os.path.join(MODEL_DIR, "pytorch_results.json")
CONFIG_FILE = os.path.join(DATA_DIR, "feature_config.json")

EXPERIMENT_NAME = "raeiphi-occupancy-classification"


def setup_mlflow():
    """Configure MLflow with SQLite backend stored inside ml/."""
    os.makedirs(MLFLOW_ARTIFACTS, exist_ok=True)
    tracking_uri = f"sqlite:///{os.path.abspath(MLFLOW_DB)}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"Artifacts dir:       {os.path.abspath(MLFLOW_ARTIFACTS)}")
    print(f"Experiment: {experiment.name} (ID: {experiment.experiment_id})")
    return experiment


def load_results():
    """Load saved results from previous training phases."""
    with open(BASELINE_RESULTS, "r") as f:
        baseline = json.load(f)

    with open(PYTORCH_RESULTS, "r") as f:
        pytorch = json.load(f)

    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)

    return baseline, pytorch, config


def log_baseline_experiment(name: str, results: dict, config: dict):
    """Log a scikit-learn baseline model run to MLflow."""
    print(f"\n  Logging: {name}...")

    with mlflow.start_run(run_name=name):
        # ── Tags ────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type": "sklearn",
            "model_name": name.lower().replace(" ", "_"),
            "phase": "1.3",
            "framework": "scikit-learn",
            "project": "raeiphi",
            "task": "occupancy_classification",
        })

        # ── Dataset info ────────────────────────────────────────
        mlflow.log_params({
            "n_features": config["n_features"],
            "n_classes": config["n_classes"],
            "train_samples": config["split"]["train_samples"],
            "val_samples": config["split"]["val_samples"],
            "test_samples": config["split"]["test_samples"],
            "scaler": config["scaler"]["type"],
        })

        # ── Hyperparameters ─────────────────────────────────────
        for key, val in results["hyperparameters"].items():
            mlflow.log_param(key, val)

        # ── Metrics ─────────────────────────────────────────────
        for key, val in results["metrics"].items():
            mlflow.log_metric(key, val)

        # Per-class F1 scores
        for i, f1 in enumerate(results["per_class_f1"]):
            mlflow.log_metric(f"f1_class_{i}", f1)

        # Training time
        mlflow.log_metric("train_time_seconds", results["train_time_seconds"])

        # ── Artifacts ───────────────────────────────────────────
        # Log confusion matrix if it exists
        cm_files = {
            "Random Forest": "08_confusion_matrix_random_forest.png",
            "Gradient Boosting": "09_confusion_matrix_gradient_boosting.png",
        }
        cm_path = os.path.join(FIG_DIR, cm_files.get(name, ""))
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, "figures")

        # Log feature importance if it exists
        fi_files = {
            "Random Forest": "11a_feature_importance_rf.png",
        }
        fi_path = os.path.join(FIG_DIR, fi_files.get(name, ""))
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path, "figures")

        # Log the model if pickle exists
        model_files = {
            "Random Forest": "random_forest.pkl",
            "Gradient Boosting": "gradient_boosting.pkl",
        }
        model_path = os.path.join(MODEL_DIR, model_files.get(name, ""))
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            mlflow.sklearn.log_model(model, "model")
            print(f"    Model artifact logged")

        print(f"    Metrics: accuracy={results['metrics']['accuracy']:.4f}, "
              f"f1_weighted={results['metrics']['f1_weighted']:.4f}")


def log_pytorch_experiment(name: str, results: dict, config: dict):
    """Log a PyTorch model run to MLflow."""
    print(f"\n  Logging: PyTorch {name}...")

    with mlflow.start_run(run_name=f"PyTorch {name}") as run:
        # ── Tags ────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type": "pytorch",
            "model_name": name.lower().replace(" ", "_"),
            "phase": "1.4",
            "framework": "pytorch",
            "project": "raeiphi",
            "task": "occupancy_classification",
            "architecture": results.get("architecture", name),
        })

        # ── Dataset info ────────────────────────────────────────
        mlflow.log_params({
            "n_features": config["n_features"],
            "n_classes": config["n_classes"],
            "train_samples": config["split"]["train_samples"],
            "val_samples": config["split"]["val_samples"],
            "test_samples": config["split"]["test_samples"],
            "scaler": config["scaler"]["type"],
        })

        # ── Architecture params ─────────────────────────────────
        mlflow.log_params({
            "architecture": results.get("architecture", name),
            "parameters": results.get("parameters", 0),
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "early_stop_patience": 15,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "lr_scheduler": "ReduceLROnPlateau",
        })

        # ── Metrics ─────────────────────────────────────────────
        for key, val in results["metrics"].items():
            if key != "per_class_f1":
                mlflow.log_metric(key, val)

        # Per-class F1 scores
        for i, f1 in enumerate(results["per_class_f1"]):
            mlflow.log_metric(f"f1_class_{i}", f1)

        # Training metadata
        mlflow.log_metric("train_time_seconds", results.get("train_time", 0))
        mlflow.log_metric("best_epoch", results.get("best_epoch", 0))
        mlflow.log_metric("total_epochs", results.get("total_epochs", 0))
        mlflow.log_metric("total_parameters", results.get("parameters", 0))

        # ── Artifacts ───────────────────────────────────────────
        # Confusion matrices
        cm_files = {
            "FCN": "14_confusion_matrix_fcn.png",
            "1D CNN": "15_confusion_matrix_cnn.png",
        }
        cm_path = os.path.join(FIG_DIR, cm_files.get(name, ""))
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, "figures")

        # Training curves
        curves_path = os.path.join(FIG_DIR, "13_training_curves.png")
        if os.path.exists(curves_path):
            mlflow.log_artifact(curves_path, "figures")

        # Comparison charts
        for fig in ["16_all_models_comparison.png", "17_per_class_f1_all_models.png"]:
            fig_path = os.path.join(FIG_DIR, fig)
            if os.path.exists(fig_path):
                mlflow.log_artifact(fig_path, "figures")

        # Log PyTorch model
        model_path = os.path.join(MODEL_DIR, "best_pytorch_model.pt")
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, "model")

        print(f"    Metrics: accuracy={results['metrics']['accuracy']:.4f}, "
              f"f1_weighted={results['metrics']['f1_weighted']:.4f}")

        return run


def log_test_set_results(test_results: dict, config: dict):
    """Log the final held-out test set evaluation."""
    print(f"\n  Logging: Test Set Evaluation...")

    with mlflow.start_run(run_name="Final Test Evaluation"):
        mlflow.set_tags({
            "model_type": "pytorch",
            "phase": "1.4",
            "evaluation_type": "held_out_test",
            "best_model": test_results["model"],
            "project": "raeiphi",
        })

        mlflow.log_params({
            "best_model": test_results["model"],
            "test_samples": config["split"]["test_samples"],
        })

        # Log test metrics with test_ prefix
        for key, val in test_results["metrics"].items():
            if key != "per_class_f1":
                mlflow.log_metric(f"test_{key}", val)

        for i, f1 in enumerate(test_results["per_class_f1"]):
            mlflow.log_metric(f"test_f1_class_{i}", f1)

        print(f"    Test accuracy: {test_results['metrics']['accuracy']:.4f}")
        print(f"    Test F1 weighted: {test_results['metrics']['f1_weighted']:.4f}")


def register_best_model(pytorch_results: dict):
    """Register the best model in MLflow Model Registry."""
    print("\n  Registering best model in Model Registry...")

    # Determine best model
    best_name = None
    best_f1 = 0
    for name in ["FCN", "1D CNN"]:
        if name in pytorch_results:
            f1 = pytorch_results[name]["metrics"]["f1_weighted"]
            if f1 > best_f1:
                best_f1 = f1
                best_name = name

    if best_name is None:
        print("    No models found to register")
        return

    model_path = os.path.join(MODEL_DIR, "best_pytorch_model.pt")
    if not os.path.exists(model_path):
        print("    Model file not found — skipping registry")
        return

    # Log a dedicated run for the registered model
    with mlflow.start_run(run_name=f"Registered — {best_name}"):
        mlflow.set_tags({
            "model_type": "pytorch",
            "phase": "1.4",
            "registered": "true",
            "architecture": best_name,
            "project": "raeiphi",
        })

        for key, val in pytorch_results[best_name]["metrics"].items():
            if key != "per_class_f1":
                mlflow.log_metric(key, val)

        # Log the model artifact
        mlflow.log_artifact(model_path, "model")

        # Log feature config alongside model for reproducibility
        mlflow.log_artifact(os.path.join(DATA_DIR, "feature_config.json"), "model")

        print(f"    Registered: {best_name}")
        print(f"    F1 weighted: {best_f1:.4f}")
        print(f"    Model + feature_config.json logged as artifacts")


def main():
    print("=" * 60)
    print("RAEIPHI — MLflow Experiment Tracking")
    print("=" * 60)

    # ── Setup ───────────────────────────────────────────────────
    experiment = setup_mlflow()

    # ── Load results ────────────────────────────────────────────
    print("\nLoading experiment results...")

    if not os.path.exists(BASELINE_RESULTS):
        print(f"  ERROR: {BASELINE_RESULTS} not found")
        print("  Run train_baseline.py first")
        sys.exit(1)

    if not os.path.exists(PYTORCH_RESULTS):
        print(f"  ERROR: {PYTORCH_RESULTS} not found")
        print("  Run train_pytorch.py first")
        sys.exit(1)

    baseline, pytorch, config = load_results()
    print(f"  Baseline models: {list(baseline.keys())}")
    print(f"  PyTorch models:  {[k for k in pytorch.keys() if k != 'test_set']}")

    # ── Log baseline experiments ────────────────────────────────
    print("\n" + "=" * 60)
    print("Logging scikit-learn baselines...")
    print("=" * 60)

    for name in ["Random Forest", "Gradient Boosting"]:
        if name in baseline:
            log_baseline_experiment(name, baseline[name], config)

    # ── Log PyTorch experiments ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Logging PyTorch experiments...")
    print("=" * 60)

    best_run = None
    for name in ["FCN", "1D CNN"]:
        if name in pytorch:
            run = log_pytorch_experiment(name, pytorch[name], config)
            if run:
                best_run = run

    # ── Log test set results ────────────────────────────────────
    if "test_set" in pytorch:
        log_test_set_results(pytorch["test_set"], config)

    # ── Register best model ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Model Registry")
    print("=" * 60)
    register_best_model(pytorch)

    # ── Summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("TRACKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Experiments logged: 6 runs total")
    print(f"    - Random Forest (sklearn baseline)")
    print(f"    - Gradient Boosting (sklearn baseline)")
    print(f"    - PyTorch FCN")
    print(f"    - PyTorch 1D CNN")
    print(f"    - Final Test Evaluation")
    print(f"    - Registered Best Model")
    print(f"\n  MLflow database: {os.path.abspath(MLFLOW_DB)}")
    print(f"\n  To view the dashboard, run:")
    print(f"    cd \"{os.path.abspath(BASE_DIR)}\"")
    print(f"    mlflow ui --backend-store-uri sqlite:///{os.path.abspath(MLFLOW_DB)} --port 5000")
    print(f"\n  Then open: http://localhost:5000")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()