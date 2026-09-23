"""
RAEIPHI — Production Model Monitor
=====================================
Analyzes the prediction log database for signs of model
degradation: confidence drift, unusual class distributions,
latency spikes, and repeated low-confidence predictions.

Usage:
  python src/monitor.py                  # Full report
  python src/monitor.py --alert-only     # Only show alerts

Input:  ml/monitoring/predictions.db
Output: Console report + ml/monitoring/monitor_report.json
        ml/notebooks/figures/18–20 (monitoring charts)
"""

import sqlite3
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MONITOR_DIR = os.path.join(BASE_DIR, "monitoring")
DB_PATH = os.path.join(MONITOR_DIR, "predictions.db")
FIG_DIR = os.path.join(BASE_DIR, "notebooks", "figures")
REPORT_PATH = os.path.join(MONITOR_DIR, "monitor_report.json")

# Alert thresholds
CONFIDENCE_FLOOR = 0.5         # Alert if avg confidence drops below this
CONFIDENCE_DRIFT_THRESHOLD = 0.1  # Alert if confidence drops >10% vs baseline
LOW_CONFIDENCE_RATE_LIMIT = 0.15  # Alert if >15% of predictions are low confidence
LATENCY_SPIKE_MS = 100.0       # Alert if avg latency exceeds this

os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="darkgrid")


def load_predictions() -> list:
    """Load all predictions from the database."""
    if not os.path.exists(DB_PATH):
        print("ERROR: No prediction database found.")
        print("Start the inference server and make some predictions first.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()

    predictions = [dict(row) for row in rows]
    if not predictions:
        print("No predictions logged yet. Run test_inference.py against the server first.")
        sys.exit(1)

    return predictions


def compute_metrics(predictions: list) -> dict:
    """Compute monitoring metrics from prediction logs."""
    confidences = [p["confidence"] for p in predictions]
    latencies = [p["inference_time_ms"] for p in predictions]
    classes = [p["predicted_class"] for p in predictions]

    # Overall stats
    metrics = {
        "total_predictions": len(predictions),
        "time_range": {
            "first": predictions[0]["timestamp"],
            "last": predictions[-1]["timestamp"],
        },
        "confidence": {
            "mean": float(np.mean(confidences)),
            "median": float(np.median(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "p5": float(np.percentile(confidences, 5)),
            "p95": float(np.percentile(confidences, 95)),
        },
        "latency": {
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "max_ms": float(np.max(latencies)),
        },
        "class_distribution": dict(Counter(classes)),
        "low_confidence_count": sum(1 for c in confidences if c < CONFIDENCE_FLOOR),
        "low_confidence_rate": sum(1 for c in confidences if c < CONFIDENCE_FLOOR) / len(confidences),
    }

    # Confidence drift: compare first half vs second half
    mid = len(confidences) // 2
    if mid > 0:
        first_half_mean = float(np.mean(confidences[:mid]))
        second_half_mean = float(np.mean(confidences[mid:]))
        drift = second_half_mean - first_half_mean
        drift_pct = (drift / first_half_mean * 100) if first_half_mean > 0 else 0

        metrics["confidence_drift"] = {
            "first_half_mean": first_half_mean,
            "second_half_mean": second_half_mean,
            "drift": float(drift),
            "drift_pct": float(drift_pct),
        }
    else:
        metrics["confidence_drift"] = None

    return metrics


def check_alerts(metrics: dict) -> list:
    """Check metrics against thresholds and return alerts."""
    alerts = []

    # Low average confidence
    if metrics["confidence"]["mean"] < CONFIDENCE_FLOOR:
        alerts.append({
            "severity": "HIGH",
            "type": "low_confidence",
            "message": f"Average confidence ({metrics['confidence']['mean']:.3f}) "
                       f"is below floor ({CONFIDENCE_FLOOR})",
            "action": "Model may need retraining with more data",
        })

    # Confidence drift
    if metrics["confidence_drift"]:
        drift = metrics["confidence_drift"]
        if abs(drift["drift_pct"]) > CONFIDENCE_DRIFT_THRESHOLD * 100:
            direction = "declining" if drift["drift"] < 0 else "increasing"
            alerts.append({
                "severity": "MEDIUM" if abs(drift["drift_pct"]) < 20 else "HIGH",
                "type": "confidence_drift",
                "message": f"Confidence is {direction}: "
                           f"{drift['first_half_mean']:.3f} → {drift['second_half_mean']:.3f} "
                           f"({drift['drift_pct']:+.1f}%)",
                "action": "Check for data distribution shift",
            })

    # High rate of low-confidence predictions
    if metrics["low_confidence_rate"] > LOW_CONFIDENCE_RATE_LIMIT:
        alerts.append({
            "severity": "MEDIUM",
            "type": "low_confidence_rate",
            "message": f"{metrics['low_confidence_rate']:.1%} of predictions have "
                       f"confidence below {CONFIDENCE_FLOOR}",
            "action": "Review input data quality or retrain on edge cases",
        })

    # Latency spike
    if metrics["latency"]["mean_ms"] > LATENCY_SPIKE_MS:
        alerts.append({
            "severity": "LOW",
            "type": "latency_spike",
            "message": f"Average latency ({metrics['latency']['mean_ms']:.1f}ms) "
                       f"exceeds threshold ({LATENCY_SPIKE_MS}ms)",
            "action": "Check server resources or model size",
        })

    # Class imbalance in predictions (might indicate systematic bias)
    class_dist = metrics["class_distribution"]
    total = metrics["total_predictions"]
    if total >= 20:
        most_common_class = max(class_dist, key=class_dist.get)
        most_common_pct = class_dist[most_common_class] / total
        if most_common_pct > 0.7:
            alerts.append({
                "severity": "MEDIUM",
                "type": "class_imbalance",
                "message": f"Class {most_common_class} accounts for "
                           f"{most_common_pct:.0%} of predictions",
                "action": "Check if input data is representative",
            })

    return alerts


def plot_confidence_over_time(predictions: list):
    """Line chart of prediction confidence over time."""
    if len(predictions) < 2:
        return

    indices = list(range(len(predictions)))
    confidences = [p["confidence"] for p in predictions]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(indices, confidences, color="#2196F3", alpha=0.6, linewidth=1)

    # Rolling average
    window = max(5, len(confidences) // 10)
    if len(confidences) >= window:
        rolling = np.convolve(confidences, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(confidences)), rolling,
                color="#FF5722", linewidth=2, label=f"Rolling avg ({window})")

    ax.axhline(y=CONFIDENCE_FLOOR, color="red", linestyle="--", alpha=0.5,
               label=f"Alert floor ({CONFIDENCE_FLOOR})")
    ax.set_xlabel("Prediction #", fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title("Model Confidence Over Time", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "18_confidence_over_time.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_class_distribution(predictions: list):
    """Bar chart of predicted class distribution."""
    classes = [p["predicted_class"] for p in predictions]
    counts = Counter(classes)

    all_classes = list(range(11))
    values = [counts.get(c, 0) for c in all_classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(all_classes, values, color=sns.color_palette("viridis", 11))
    ax.set_xlabel("Predicted Occupancy", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xticks(all_classes)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "19_prediction_class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_latency_distribution(predictions: list):
    """Histogram of inference latency."""
    latencies = [p["inference_time_ms"] for p in predictions]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(latencies, bins=30, color="#4CAF50", alpha=0.7, edgecolor="black")
    ax.axvline(x=np.mean(latencies), color="red", linestyle="--",
               label=f"Mean: {np.mean(latencies):.1f}ms")
    ax.axvline(x=np.percentile(latencies, 95), color="orange", linestyle="--",
               label=f"P95: {np.percentile(latencies, 95):.1f}ms")
    ax.set_xlabel("Inference Time (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Inference Latency Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "20_latency_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_report(metrics: dict, alerts: list):
    """Print the monitoring report to console."""
    print(f"\n{'=' * 60}")
    print("RAEIPHI — Production Model Monitor")
    print(f"{'=' * 60}")

    print(f"\n  Total predictions: {metrics['total_predictions']:,}")
    print(f"  Time range: {metrics['time_range']['first'][:19]}")
    print(f"            → {metrics['time_range']['last'][:19]}")

    print(f"\n  CONFIDENCE")
    c = metrics["confidence"]
    print(f"    Mean:    {c['mean']:.4f}")
    print(f"    Median:  {c['median']:.4f}")
    print(f"    Std:     {c['std']:.4f}")
    print(f"    Range:   [{c['min']:.4f}, {c['max']:.4f}]")
    print(f"    P5–P95:  [{c['p5']:.4f}, {c['p95']:.4f}]")

    if metrics["confidence_drift"]:
        d = metrics["confidence_drift"]
        print(f"\n  CONFIDENCE DRIFT")
        print(f"    First half avg:  {d['first_half_mean']:.4f}")
        print(f"    Second half avg: {d['second_half_mean']:.4f}")
        print(f"    Drift:           {d['drift']:+.4f} ({d['drift_pct']:+.1f}%)")

    print(f"\n  LATENCY")
    l = metrics["latency"]
    print(f"    Mean:    {l['mean_ms']:.1f}ms")
    print(f"    Median:  {l['median_ms']:.1f}ms")
    print(f"    P95:     {l['p95_ms']:.1f}ms")
    print(f"    Max:     {l['max_ms']:.1f}ms")

    print(f"\n  LOW CONFIDENCE PREDICTIONS")
    print(f"    Count:   {metrics['low_confidence_count']}")
    print(f"    Rate:    {metrics['low_confidence_rate']:.1%}")

    print(f"\n  CLASS DISTRIBUTION")
    for cls in sorted(metrics["class_distribution"].keys()):
        count = metrics["class_distribution"][cls]
        pct = count / metrics["total_predictions"] * 100
        bar = "█" * int(pct * 2)
        print(f"    Class {cls:>2}: {count:>5} ({pct:>5.1f}%)  {bar}")

    # Alerts
    print(f"\n{'=' * 60}")
    if alerts:
        print(f"  ALERTS ({len(alerts)})")
        print(f"{'=' * 60}")
        for alert in alerts:
            severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                alert["severity"], "⚪"
            )
            print(f"\n  {severity_icon} [{alert['severity']}] {alert['type']}")
            print(f"     {alert['message']}")
            print(f"     Action: {alert['action']}")
    else:
        print("  NO ALERTS — model is healthy")
        print(f"{'=' * 60}")


def main():
    alert_only = "--alert-only" in sys.argv

    # Load data
    predictions = load_predictions()

    # Compute metrics
    metrics = compute_metrics(predictions)

    # Check alerts
    alerts = check_alerts(metrics)

    if alert_only:
        if alerts:
            for alert in alerts:
                print(f"[{alert['severity']}] {alert['type']}: {alert['message']}")
        else:
            print("OK — no alerts")
        return

    # Full report
    print_report(metrics, alerts)

    # Generate charts
    print(f"\n  Generating monitoring charts...")
    plot_confidence_over_time(predictions)
    plot_class_distribution(predictions)
    plot_latency_distribution(predictions)

    # Save report JSON
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "alerts": alerts,
    }
    # Convert int keys to strings for JSON
    report["metrics"]["class_distribution"] = {
        str(k): v for k, v in report["metrics"]["class_distribution"].items()
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {REPORT_PATH}")

    print(f"\n{'=' * 60}")
    print("MONITORING COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
