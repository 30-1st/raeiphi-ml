"""
RAEIPHI — Leak Detection Inference Server
============================================
FastAPI endpoint that runs both the autoencoder (anomaly score)
and binary classifier (leak probability) on incoming CSI data.

Endpoints:
  GET  /health          — server + model status
  POST /detect          — single CSI reading → leak detection
  POST /detect/batch    — multiple readings → batch detection
  GET  /model/info      — model metadata

Runs alongside the occupancy inference server on a different port.

Run:  python src/leak_inference_server.py
      (starts on port 8002)
"""

import json
import os
import time
import logging
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction_logger import PredictionLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("raeiphi-leak")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
AE_PATH = os.path.join(BASE_DIR, "models", "leak_autoencoder.keras")
CLF_PATH = os.path.join(BASE_DIR, "models", "leak_classifier.keras")
CONFIG_PATH = os.path.join(BASE_DIR, "data", "leak_feature_config.json")


# ── Feature Engineering (mirrors leak pipeline) ─────────────────
def engineer_leak_features(raw: dict, n_sub: int = 52) -> dict:
    """Apply the same feature engineering as leak_feature_engineering.py."""
    features = dict(raw)

    # SNR per subcarrier
    for i in range(n_sub):
        features[f"snr_sc{i}"] = raw.get(f"amp_mean_sc{i}", 0) / (raw.get(f"amp_std_sc{i}", 0) + 1e-6)

    snr_vals = [features[f"snr_sc{i}"] for i in range(n_sub)]
    features["snr_min"] = float(np.min(snr_vals))
    features["snr_max"] = float(np.max(snr_vals))

    amp_std_vals = [raw.get(f"amp_std_sc{i}", 0) for i in range(n_sub)]
    amp_mean_vals = [raw.get(f"amp_mean_sc{i}", 0) for i in range(n_sub)]

    features["amp_std_skew"] = float(_skew(amp_std_vals))
    features["amp_std_kurtosis"] = float(_kurtosis(amp_std_vals))
    features["amp_mean_skew"] = float(_skew(amp_mean_vals))
    features["amp_mean_kurtosis"] = float(_kurtosis(amp_mean_vals))

    for bname, idxs in [("lower", range(0, 17)), ("mid", range(17, 35)), ("upper", range(35, 52))]:
        features[f"amp_std_{bname}_mean"] = float(np.mean([amp_std_vals[i] for i in idxs]))
        features[f"amp_mean_{bname}_mean"] = float(np.mean([amp_mean_vals[i] for i in idxs]))

    features["band_amplitude_variance"] = float(np.std([
        features["amp_mean_lower_mean"], features["amp_mean_mid_mean"], features["amp_mean_upper_mean"]
    ]))

    features["amp_std_p10"] = float(np.percentile(amp_std_vals, 10))
    features["amp_std_p50"] = float(np.percentile(amp_std_vals, 50))
    features["amp_std_p90"] = float(np.percentile(amp_std_vals, 90))
    features["amp_std_iqr"] = features["amp_std_p90"] - features["amp_std_p10"]

    features["phase_to_temporal_ratio"] = raw.get("phase_std_global", 0) / (raw.get("temporal_diff_mean", 0) + 1e-6)
    features["drop_to_variance_ratio"] = raw.get("amplitude_drop_magnitude", 0) / (raw.get("amp_std_global", 0) + 1e-6)

    return features


def _skew(values):
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 3: return 0.0
    mean, std = np.mean(arr), np.std(arr, ddof=1)
    return 0.0 if std == 0 else float((n / ((n-1)*(n-2))) * np.sum(((arr - mean) / std) ** 3))


def _kurtosis(values):
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n < 4: return 0.0
    mean, std = np.mean(arr), np.std(arr, ddof=1)
    return 0.0 if std == 0 else float(np.mean((arr - mean)**4) / (std**4) - 3.0)


# ── Global State ────────────────────────────────────────────────
state = {
    "autoencoder": None,
    "classifier": None,
    "config": None,
    "ae_threshold": None,
    "loaded": False,
    "request_count": 0,
    "pred_logger": None,
}


def load_models():
    logger.info("Loading leak detection models...")
    start = time.time()

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    ae = tf.keras.models.load_model(AE_PATH)
    clf = tf.keras.models.load_model(CLF_PATH)

    # Load threshold from results
    results_path = os.path.join(BASE_DIR, "models", "leak_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        threshold = results.get("Autoencoder", {}).get("threshold", 0.01)
    else:
        threshold = 0.01

    lt = time.time() - start
    state.update(
        autoencoder=ae, classifier=clf, config=config,
        ae_threshold=threshold, loaded=True,
        pred_logger=PredictionLogger(
            db_path=os.path.join(BASE_DIR, "monitoring", "leak_predictions.db")
        ),
    )
    logger.info(f"Models loaded in {lt:.2f}s — {config['n_features']} features")
    logger.info(f"Autoencoder threshold: {threshold:.6f}")
    logger.info("Leak prediction logger initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    logger.info("Shutting down leak detection server")


app = FastAPI(
    title="RAEIPHI Leak Detection API",
    description="Water/leak detection from WiFi CSI data",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ─────────────────────────────────────────────────────
class CSIReading(BaseModel):
    amp_mean: list[float] = Field(..., description="52 amplitude means")
    amp_std: list[float] = Field(..., description="52 amplitude std devs")
    phase_std: list[float] = Field(..., description="52 phase std devs")
    amp_mean_global: float
    amp_std_global: float
    amp_range_global: float
    amp_std_max: float
    amp_std_min: float
    amp_variance_spread: float
    phase_std_global: float
    phase_std_max: float
    phase_std_spread: float
    subcarrier_corr_mean: float
    subcarrier_corr_std: float
    temporal_diff_mean: float
    temporal_diff_max: float
    temporal_diff_std: float
    # Water-specific features from the sensor
    absorption_uniformity: float = 0.0
    temporal_stability_ratio: float = 0.0
    broadband_absorption_score: float = 0.0
    amplitude_drop_magnitude: float = 0.0
    phase_drift_magnitude: float = 0.0
    phase_drift_consistency: float = 0.0
    snr_global_mean: float = 0.0
    snr_global_std: float = 0.0
    max_contiguous_affected: int = 0
    contiguity_ratio: float = 0.0


class LeakDetectionResponse(BaseModel):
    leak_detected: bool = Field(..., description="Whether a leak/water is detected")
    leak_probability: float = Field(..., description="Classifier confidence (0-1)")
    anomaly_score: float = Field(..., description="Autoencoder reconstruction error")
    anomaly_detected: bool = Field(..., description="Whether anomaly exceeds threshold")
    severity: str = Field(..., description="none, low, medium, high")
    inference_time_ms: float


class BatchLeakResponse(BaseModel):
    detections: list[LeakDetectionResponse]
    total_time_ms: float
    count: int


# ── Helpers ─────────────────────────────────────────────────────
def reading_to_vector(reading: CSIReading) -> np.ndarray:
    config = state["config"]
    fc = config["feature_columns"]
    sm = config["scaler"]["means"]
    ss = config["scaler"]["stds"]

    raw = {}
    for i in range(52):
        raw[f"amp_mean_sc{i}"] = reading.amp_mean[i]
        raw[f"amp_std_sc{i}"] = reading.amp_std[i]
        raw[f"phase_std_sc{i}"] = reading.phase_std[i]

    for k in ["amp_mean_global", "amp_std_global", "amp_range_global", "amp_std_max",
              "amp_std_min", "amp_variance_spread", "phase_std_global", "phase_std_max",
              "phase_std_spread", "subcarrier_corr_mean", "subcarrier_corr_std",
              "temporal_diff_mean", "temporal_diff_max", "temporal_diff_std",
              "absorption_uniformity", "temporal_stability_ratio",
              "broadband_absorption_score", "amplitude_drop_magnitude",
              "phase_drift_magnitude", "phase_drift_consistency",
              "snr_global_mean", "snr_global_std",
              "max_contiguous_affected", "contiguity_ratio"]:
        raw[k] = getattr(reading, k, 0)

    features = engineer_leak_features(raw)
    return np.array([(features.get(c, 0.0) - sm.get(c, 0.0)) / ss.get(c, 1.0) for c in fc], dtype=np.float32)


def run_detection(vector: np.ndarray) -> dict:
    ae = state["autoencoder"]
    clf = state["classifier"]
    threshold = state["ae_threshold"]

    v = vector.reshape(1, -1)

    # Autoencoder anomaly score
    reconstructed = ae.predict(v, verbose=0)
    anomaly_score = float(np.mean(np.square(v - reconstructed)))
    anomaly_detected = anomaly_score > threshold

    # Classifier probability
    leak_prob = float(clf.predict(v, verbose=0).flatten()[0])
    leak_detected = leak_prob > 0.5

    # Severity
    if not leak_detected:
        severity = "none"
    elif leak_prob < 0.6:
        severity = "low"
    elif leak_prob < 0.85:
        severity = "medium"
    else:
        severity = "high"

    return {
        "leak_detected": leak_detected,
        "leak_probability": round(leak_prob, 4),
        "anomaly_score": round(anomaly_score, 6),
        "anomaly_detected": anomaly_detected,
        "severity": severity,
    }


# ── Endpoints ───────────────────────────────────────────────────
@app.get("/health")
async def health():
    pl = state.get("pred_logger")
    return {
        "status": "healthy" if state["loaded"] else "loading",
        "models_loaded": state["loaded"],
        "ae_threshold": state["ae_threshold"],
        "predictions_logged": pl.get_total_predictions() if pl else 0,
    }


@app.get("/model/info")
async def model_info():
    if not state["loaded"]: raise HTTPException(503, "Models not loaded")
    config = state["config"]
    return {
        "autoencoder": "Dense(128→64→32→64→128)",
        "classifier": "Dense(128→64→32→1, sigmoid)",
        "framework": "tensorflow",
        "n_features": config["n_features"],
        "ae_threshold": state["ae_threshold"],
    }


@app.post("/detect", response_model=LeakDetectionResponse)
async def detect(reading: CSIReading):
    if not state["loaded"]: raise HTTPException(503, "Models not loaded")
    if len(reading.amp_mean) != 52: raise HTTPException(422, "Need 52 subcarrier values")

    start = time.time()
    vector = reading_to_vector(reading)
    result = run_detection(vector)
    t_ms = (time.time() - start) * 1000
    state["request_count"] += 1

    pl = state["pred_logger"]
    if pl:
        pl.log(
            input_hash=pl.hash_input(vector),
            predicted_class=1 if result["leak_detected"] else 0,
            confidence=result["leak_probability"],
            inference_time_ms=t_ms,
            top_3_classes={"dry": round(1 - result["leak_probability"], 4),
                          "wet": result["leak_probability"]},
        )

    return LeakDetectionResponse(**result, inference_time_ms=round(t_ms, 2))


@app.post("/detect/batch", response_model=BatchLeakResponse)
async def detect_batch(readings: list[CSIReading]):
    if not state["loaded"]: raise HTTPException(503, "Models not loaded")
    if len(readings) > 100: raise HTTPException(422, "Max 100 per batch")

    start = time.time()
    detections = []

    for reading in readings:
        rs = time.time()
        vector = reading_to_vector(reading)
        result = run_detection(vector)
        rt = (time.time() - rs) * 1000

        pl = state["pred_logger"]
        if pl:
            pl.log(
                input_hash=pl.hash_input(vector),
                predicted_class=1 if result["leak_detected"] else 0,
                confidence=result["leak_probability"],
                inference_time_ms=rt,
            )

        detections.append(LeakDetectionResponse(**result, inference_time_ms=round(rt, 2)))

    state["request_count"] += len(readings)
    return BatchLeakResponse(
        detections=detections,
        total_time_ms=round((time.time() - start) * 1000, 2),
        count=len(detections),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
