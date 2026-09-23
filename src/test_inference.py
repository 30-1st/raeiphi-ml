"""
RAEIPHI — Inference Server Test
================================
Generates sample CSI readings and sends them to the inference
server to verify predictions are returned correctly.

Usage:
  1. Start the server:  uvicorn src.inference_server:app --port 8000
  2. Run this script:   python src/test_inference.py

Tests:
  - Health check
  - Model info
  - Single prediction (empty room)
  - Single prediction (occupied room)
  - Batch prediction (multiple occupancy levels)
"""

import json
import sys
import numpy as np
import requests

SERVER = "http://localhost:8000"
np.random.seed(42)


def generate_sample_reading(occupancy: int) -> dict:
    """
    Generate a synthetic CSI reading for a given occupancy level.
    Mirrors the logic from generate_dataset.py.
    """
    n_sub = 52
    n_time = 10

    # Base signal
    base = np.random.uniform(20, 40, size=n_sub)
    edge_rolloff = np.concatenate([
        np.linspace(0.7, 1.0, n_sub // 4),
        np.ones(n_sub // 2),
        np.linspace(1.0, 0.7, n_sub - n_sub // 4 - n_sub // 2)
    ])
    base_signal = base * edge_rolloff

    # Amplitude
    amp_variance_scale = 0.5 + 1.8 * np.sqrt(occupancy)
    absorption_factor = 1.0 - (occupancy * 0.015)

    snapshots = np.zeros((n_time, n_sub))
    for t in range(n_time):
        noise = np.random.normal(0, amp_variance_scale, size=n_sub)
        if occupancy > 0:
            n_affected = min(n_sub, int(n_sub * 0.3 * np.sqrt(occupancy)))
            affected = np.random.choice(n_sub, n_affected, replace=False)
            noise[affected] *= np.random.uniform(1.5, 3.0, size=n_affected)
        snapshots[t] = base_signal * absorption_factor + noise

    amp_mean = np.mean(snapshots, axis=0).tolist()
    amp_std = np.std(snapshots, axis=0).tolist()
    amp_range = (np.max(snapshots, axis=0) - np.min(snapshots, axis=0)).tolist()

    # Phase
    base_phase = np.random.uniform(-np.pi, np.pi, size=n_sub)
    phase_instability = 0.05 + 0.25 * np.sqrt(occupancy)
    phase_snapshots = np.zeros((n_time, n_sub))
    for t in range(n_time):
        phase_snapshots[t] = base_phase + np.random.normal(0, phase_instability, size=n_sub)
    phase_std = np.std(phase_snapshots, axis=0).tolist()

    # Temporal dynamics
    temporal_diffs = np.diff(snapshots, axis=0)

    # Subcarrier correlation
    corr_matrix = np.corrcoef(snapshots.T)
    upper_tri = corr_matrix[np.triu_indices(n_sub, k=1)]

    return {
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "phase_std": phase_std,
        "amp_mean_global": float(np.mean(amp_mean)),
        "amp_std_global": float(np.mean(amp_std)),
        "amp_range_global": float(np.mean(amp_range)),
        "amp_std_max": float(np.max(amp_std)),
        "amp_std_min": float(np.min(amp_std)),
        "amp_variance_spread": float(np.max(amp_std) - np.min(amp_std)),
        "phase_std_global": float(np.mean(phase_std)),
        "phase_std_max": float(np.max(phase_std)),
        "phase_std_spread": float(np.max(phase_std) - np.min(phase_std)),
        "subcarrier_corr_mean": float(np.mean(upper_tri)),
        "subcarrier_corr_std": float(np.std(upper_tri)),
        "temporal_diff_mean": float(np.mean(np.abs(temporal_diffs))),
        "temporal_diff_max": float(np.max(np.abs(temporal_diffs))),
        "temporal_diff_std": float(np.std(temporal_diffs)),
    }


def test_health():
    """Test health endpoint."""
    print("─" * 50)
    print("TEST: Health Check")
    print("─" * 50)
    resp = requests.get(f"{SERVER}/health")
    data = resp.json()
    print(f"  Status:       {data['status']}")
    print(f"  Model loaded: {data['model_loaded']}")
    print(f"  Architecture: {data.get('model_architecture', 'N/A')}")
    assert data["status"] == "healthy", "Server not healthy"
    assert data["model_loaded"], "Model not loaded"
    print("  ✓ PASSED\n")


def test_model_info():
    """Test model info endpoint."""
    print("─" * 50)
    print("TEST: Model Info")
    print("─" * 50)
    resp = requests.get(f"{SERVER}/model/info")
    data = resp.json()
    print(f"  Architecture: {data['architecture']}")
    print(f"  Features:     {data['n_features']}")
    print(f"  Classes:      {data['n_classes']}")
    print(f"  Model size:   {data['model_size_kb']:.0f} KB")
    print(f"  F1 weighted:  {data['metrics']['f1_weighted']:.4f}")
    print("  ✓ PASSED\n")


def test_single_prediction(occupancy: int, label: str):
    """Test single prediction endpoint."""
    print("─" * 50)
    print(f"TEST: Single Prediction — {label} (actual: {occupancy})")
    print("─" * 50)

    reading = generate_sample_reading(occupancy)
    resp = requests.post(f"{SERVER}/predict", json=reading)

    if resp.status_code != 200:
        print(f"  ✗ FAILED — HTTP {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    predicted = data["occupancy"]
    confidence = data["confidence"]
    inference_ms = data["inference_time_ms"]

    # Show top 3 probabilities
    probs = sorted(data["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
    top3 = ", ".join([f"{k}: {v:.3f}" for k, v in probs])

    print(f"  Predicted:    {predicted}")
    print(f"  Confidence:   {confidence:.4f}")
    print(f"  Top 3 probs:  {top3}")
    print(f"  Inference:    {inference_ms:.2f}ms")

    close_enough = abs(predicted - occupancy) <= 1
    print(f"  Within ±1:    {'✓' if close_enough else '✗'}")
    print(f"  ✓ PASSED\n")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("─" * 50)
    print("TEST: Batch Prediction (0, 2, 5, 8 occupants)")
    print("─" * 50)

    readings = [generate_sample_reading(occ) for occ in [0, 2, 5, 8]]
    resp = requests.post(f"{SERVER}/predict/batch", json=readings)

    if resp.status_code != 200:
        print(f"  ✗ FAILED — HTTP {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    print(f"  Batch size:    {data['count']}")
    print(f"  Total time:    {data['total_inference_time_ms']:.2f}ms")
    print()

    actuals = [0, 2, 5, 8]
    for i, pred in enumerate(data["predictions"]):
        status = "✓" if abs(pred["occupancy"] - actuals[i]) <= 1 else "~"
        print(f"  Actual: {actuals[i]:>2}  →  Predicted: {pred['occupancy']:>2}  "
              f"(conf: {pred['confidence']:.3f}, {pred['inference_time_ms']:.1f}ms)  {status}")

    print(f"\n  ✓ PASSED\n")


def main():
    print("=" * 50)
    print("RAEIPHI — Inference Server Tests")
    print("=" * 50)
    print(f"Server: {SERVER}\n")

    # Check server is running
    try:
        requests.get(f"{SERVER}/health", timeout=3)
    except requests.ConnectionError:
        print("ERROR: Server not running.")
        print("Start it first:")
        print("  cd ml")
        print("  uvicorn src.inference_server:app --port 8000")
        sys.exit(1)

    test_health()
    test_model_info()
    test_single_prediction(0, "Empty room")
    test_single_prediction(2, "Couple")
    test_single_prediction(7, "Large group")
    test_batch_prediction()

    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
