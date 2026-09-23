"""
RAEIPHI — Occupancy Classification Inference Server v1.1
=========================================================
Now with production monitoring — every prediction is logged.

Endpoints:
  GET  /health          — server + model status + prediction count
  POST /predict          — single CSI reading → occupancy prediction
  POST /predict/batch    — multiple CSI readings → batch predictions
  GET  /model/info       — model metadata and performance metrics
  GET  /monitor/recent   — recent prediction logs
  GET  /monitor/stats    — confidence and latency statistics

Run:  python src/inference_server.py
"""

import json, os, time, logging
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction_logger import PredictionLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("raeiphi-inference")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_pytorch_model.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "data", "feature_config.json")


# ── Model Architectures ────────────────────────────────────────
class OccupancyCNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.fc_layers(self.conv_layers(x.unsqueeze(1)))

class OccupancyFCN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):
        return self.network(x)


# ── Feature Engineering ─────────────────────────────────────────
def _skew(values):
    arr = np.array(values, dtype=float); n = len(arr)
    if n < 3: return 0.0
    mean, std = np.mean(arr), np.std(arr, ddof=1)
    return 0.0 if std == 0 else float((n/((n-1)*(n-2)))*np.sum(((arr-mean)/std)**3))

def _kurtosis(values):
    arr = np.array(values, dtype=float); n = len(arr)
    if n < 4: return 0.0
    mean, std = np.mean(arr), np.std(arr, ddof=1)
    return 0.0 if std == 0 else float(np.mean((arr-mean)**4)/(std**4) - 3.0)

def engineer_features_single(raw, n_subcarriers=52):
    features = dict(raw)
    for i in range(n_subcarriers):
        features[f"snr_sc{i}"] = raw.get(f"amp_mean_sc{i}",0)/(raw.get(f"amp_std_sc{i}",0)+1e-6)
    snr_vals = [features[f"snr_sc{i}"] for i in range(n_subcarriers)]
    features["snr_global_mean"] = float(np.mean(snr_vals))
    features["snr_global_std"] = float(np.std(snr_vals))
    features["snr_global_min"] = float(np.min(snr_vals))
    features["snr_global_max"] = float(np.max(snr_vals))
    amp_means = [raw.get(f"amp_mean_sc{i}",0) for i in range(n_subcarriers)]
    amp_stds = [raw.get(f"amp_std_sc{i}",0) for i in range(n_subcarriers)]
    features["amp_mean_skew"] = float(_skew(amp_means))
    features["amp_mean_kurtosis"] = float(_kurtosis(amp_means))
    features["amp_std_skew"] = float(_skew(amp_stds))
    features["amp_std_kurtosis"] = float(_kurtosis(amp_stds))
    phase_stds = [raw.get(f"phase_std_sc{i}",0) for i in range(n_subcarriers)]
    features["phase_std_skew"] = float(_skew(phase_stds))
    features["phase_std_kurtosis"] = float(_kurtosis(phase_stds))
    for bname, idxs in [("lower",range(0,17)),("mid",range(17,35)),("upper",range(35,52))]:
        ba = [amp_stds[i] for i in idxs]; bp = [phase_stds[i] for i in idxs]
        features[f"amp_std_{bname}_mean"] = float(np.mean(ba))
        features[f"amp_std_{bname}_std"] = float(np.std(ba))
        features[f"phase_std_{bname}_mean"] = float(np.mean(bp))
    features["amp_std_band_contrast"] = features["amp_std_upper_mean"]-features["amp_std_lower_mean"]
    features["phase_std_band_contrast"] = features["phase_std_upper_mean"]-features["phase_std_lower_mean"]
    features["amp_std_p25"] = float(np.percentile(amp_stds, 25))
    features["amp_std_p50"] = float(np.percentile(amp_stds, 50))
    features["amp_std_p75"] = float(np.percentile(amp_stds, 75))
    features["amp_std_iqr"] = features["amp_std_p75"]-features["amp_std_p25"]
    features["phase_to_amp_ratio"] = raw.get("phase_std_global",0)/(raw.get("amp_std_global",0)+1e-6)
    features["temporal_to_amp_ratio"] = raw.get("temporal_diff_mean",0)/(raw.get("amp_std_global",0)+1e-6)
    features["range_to_mean_ratio"] = raw.get("amp_range_global",0)/(raw.get("amp_mean_global",0)+1e-6)
    return features


# ── Global State ────────────────────────────────────────────────
model_state = {"model":None,"config":None,"checkpoint":None,"loaded":False,"load_time":None,"request_count":0,"pred_logger":None}

def load_model():
    logger.info(f"Loading model from {MODEL_PATH}")
    start = time.time()
    with open(CONFIG_PATH) as f: config = json.load(f)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    n_feat, n_cls, arch = checkpoint["n_features"], checkpoint["n_classes"], checkpoint["architecture"]
    model = OccupancyCNN(n_feat, n_cls) if arch == "1D CNN" else OccupancyFCN(n_feat, n_cls)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    lt = time.time()-start
    model_state.update(model=model, config=config, checkpoint=checkpoint, loaded=True, load_time=lt, pred_logger=PredictionLogger())
    logger.info(f"Model loaded in {lt:.2f}s — {arch}, {n_feat} features, {n_cls} classes")
    logger.info("Prediction logger initialized")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(); yield; logger.info("Shutting down")

app = FastAPI(title="RAEIPHI Occupancy Inference API", description="Real-time occupancy prediction with production monitoring", version="1.1.0", lifespan=lifespan)


# ── Schemas ─────────────────────────────────────────────────────
class CSIReading(BaseModel):
    amp_mean: list[float]; amp_std: list[float]; phase_std: list[float]
    amp_mean_global: float; amp_std_global: float; amp_range_global: float
    amp_std_max: float; amp_std_min: float; amp_variance_spread: float
    phase_std_global: float; phase_std_max: float; phase_std_spread: float
    subcarrier_corr_mean: float; subcarrier_corr_std: float
    temporal_diff_mean: float; temporal_diff_max: float; temporal_diff_std: float

class PredictionResponse(BaseModel):
    occupancy: int; confidence: float; probabilities: dict[str,float]; inference_time_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]; total_inference_time_ms: float; count: int


# ── Helpers ─────────────────────────────────────────────────────
def reading_to_feature_vector(reading):
    config = model_state["config"]; fc = config["feature_columns"]
    sm, ss = config["scaler"]["means"], config["scaler"]["stds"]
    raw = {}
    for i in range(52):
        raw[f"amp_mean_sc{i}"]=reading.amp_mean[i]; raw[f"amp_std_sc{i}"]=reading.amp_std[i]; raw[f"phase_std_sc{i}"]=reading.phase_std[i]
    for k in ["amp_mean_global","amp_std_global","amp_range_global","amp_std_max","amp_std_min","amp_variance_spread","phase_std_global","phase_std_max","phase_std_spread","subcarrier_corr_mean","subcarrier_corr_std","temporal_diff_mean","temporal_diff_max","temporal_diff_std"]:
        raw[k] = getattr(reading, k)
    features = engineer_features_single(raw)
    return np.array([(features.get(c,0.0)-sm.get(c,0.0))/ss.get(c,1.0) for c in fc], dtype=np.float32)

def run_inference(fv):
    with torch.no_grad():
        out = model_state["model"](torch.FloatTensor(fv).unsqueeze(0))
        probs = torch.softmax(out, dim=1); conf, pred = torch.max(probs, 1)
    return int(pred.item()), float(conf.item()), {str(i):round(float(p),4) for i,p in enumerate(probs[0])}


# ── Endpoints ───────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    pl = model_state.get("pred_logger")
    return {"status":"healthy" if model_state["loaded"] else "loading", "model_loaded":model_state["loaded"],
            "model_architecture":model_state["checkpoint"].get("architecture") if model_state["loaded"] else None,
            "uptime_requests":model_state["request_count"], "predictions_logged":pl.get_total_predictions() if pl else 0}

@app.get("/model/info")
async def model_info():
    if not model_state["loaded"]: raise HTTPException(503, "Model not loaded")
    cp = model_state["checkpoint"]
    return {"architecture":cp["architecture"],"n_features":cp["n_features"],"n_classes":cp["n_classes"],
            "classes":cp["classes"],"metrics":cp["metrics"],"model_size_kb":round(os.path.getsize(MODEL_PATH)/1024,1)}

@app.post("/predict", response_model=PredictionResponse)
async def predict(reading: CSIReading):
    if not model_state["loaded"]: raise HTTPException(503, "Model not loaded")
    if len(reading.amp_mean)!=52 or len(reading.amp_std)!=52 or len(reading.phase_std)!=52:
        raise HTTPException(422, "Each subcarrier array must have exactly 52 values")
    start = time.time()
    fv = reading_to_feature_vector(reading)
    occ, conf, probs = run_inference(fv)
    t_ms = (time.time()-start)*1000
    model_state["request_count"] += 1
    pl = model_state["pred_logger"]
    if pl:
        top3 = dict(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:3])
        pl.log(input_hash=pl.hash_input(fv), predicted_class=occ, confidence=conf, inference_time_ms=t_ms, top_3_classes=top3)
    return PredictionResponse(occupancy=occ, confidence=round(conf,4), probabilities=probs, inference_time_ms=round(t_ms,2))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(readings: list[CSIReading]):
    if not model_state["loaded"]: raise HTTPException(503, "Model not loaded")
    if len(readings)>100: raise HTTPException(422, "Max 100 readings per batch")
    start = time.time(); preds = []
    for r in readings:
        if len(r.amp_mean)!=52 or len(r.amp_std)!=52 or len(r.phase_std)!=52:
            raise HTTPException(422, "Each subcarrier array must have exactly 52 values")
        rs = time.time(); fv = reading_to_feature_vector(r); occ, conf, probs = run_inference(fv); rt = (time.time()-rs)*1000
        pl = model_state["pred_logger"]
        if pl:
            top3 = dict(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:3])
            pl.log(input_hash=pl.hash_input(fv), predicted_class=occ, confidence=conf, inference_time_ms=rt, top_3_classes=top3)
        preds.append(PredictionResponse(occupancy=occ, confidence=round(conf,4), probabilities=probs, inference_time_ms=round(rt,2)))
    model_state["request_count"] += len(readings)
    return BatchPredictionResponse(predictions=preds, total_inference_time_ms=round((time.time()-start)*1000,2), count=len(preds))

@app.get("/monitor/recent")
async def monitor_recent():
    pl = model_state.get("pred_logger")
    if not pl: raise HTTPException(503, "Logger not initialized")
    return {"predictions": pl.get_recent(20)}

@app.get("/monitor/stats")
async def monitor_stats():
    pl = model_state.get("pred_logger")
    if not pl: raise HTTPException(503, "Logger not initialized")
    total = pl.get_total_predictions(); recent = pl.get_recent(100)
    if not recent: return {"total":0, "message":"No predictions logged yet"}
    confs = [p["confidence"] for p in recent]; lats = [p["inference_time_ms"] for p in recent]
    return {"total_predictions":total, "recent_sample_size":len(recent),
            "confidence":{"mean":round(float(np.mean(confs)),4),"min":round(float(np.min(confs)),4),"max":round(float(np.max(confs)),4)},
            "latency":{"mean_ms":round(float(np.mean(lats)),2),"p95_ms":round(float(np.percentile(lats,95)),2)}}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
