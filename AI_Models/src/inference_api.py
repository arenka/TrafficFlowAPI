"""
FastAPI AI Inference Microservice for METR-LA Spatio-Temporal Traffic Flow Forecasting.
Exposes trained PyTorch models (LSTM, GRU, BiLSTM-Attention) to the ASP.NET Core Web API.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    CONGESTION_MODERATE_MPH,
    CONGESTION_SEVERE_MPH,
    HORIZONS,
    INPUT_SEQ_LEN,
    MODELS_DIR,
    RESULTS_DIR,
)
from data_loader import (
    build_spatial_weight_matrix,
    load_metr_la_dataframe,
    load_sensor_locations,
    prepare_spatiotemporal_arrays,
)
from models import build_model
from weather_analysis import align_weather_to_metr_la, wmo_code_to_tr, WEATHER_PATH

STATE: dict = {}


def classify_congestion(speed_mph: float) -> str:
    if speed_mph < CONGESTION_SEVERE_MPH:
        return "Severe"
    if speed_mph < CONGESTION_MODERATE_MPH:
        return "Moderate"
    return "FreeFlow"


def load_trained_models():
    models = {}
    device = torch.device("cpu")
    mapping = {
        "lstm": MODELS_DIR / "lstm_best.pth",
        "gru": MODELS_DIR / "gru_best.pth",
        "bilstm_attention": MODELS_DIR / "bilstm_attention_best.pth",
    }
    for key, path in mapping.items():
        if not path.exists():
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = build_model(
            model_type=ckpt["model_type"],
            input_dim=cfg["input_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            num_horizons=cfg["num_horizons"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models[key] = {
            "model": model,
            "mean": float(ckpt["scaler"]["mean"]),
            "std": float(ckpt["scaler"]["std"]),
            "metrics": ckpt.get("metrics", {}),
        }
    return models


@asynccontextmanager
async def lifespan(app: FastAPI):
    df = load_metr_la_dataframe()
    sensor_ids = list(df.columns)
    adj_norm = build_spatial_weight_matrix(sensor_ids)
    data = prepare_spatiotemporal_arrays(df, adj_norm)
    locations = load_sensor_locations(sensor_ids)
    models = load_trained_models()

    # Load experiment results if available
    exp_path = RESULTS_DIR / "experiment_results.json"
    experiment_results = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.exists() else {}

    STATE["data"] = data
    STATE["adj_norm"] = adj_norm
    STATE["locations"] = locations
    STATE["sensor_index_map"] = {sid: idx for idx, sid in enumerate(sensor_ids)}
    STATE["models"] = models
    STATE["experiments"] = experiment_results
    STATE["weather"] = align_weather_to_metr_la(data["timestamps"]) if WEATHER_PATH.exists() else None
    yield
    STATE.clear()


class CustomPredictRequest(BaseModel):
    speeds: list[float]
    neighborSpeeds: list[float] | None = None
    hour: int = 8
    minute: int = 30
    dayOfWeek: int = 2
    model: Literal["lstm", "gru", "bilstm_attention"] = "gru"


app = FastAPI(
    title="METR-LA Traffic Flow AI Inference Microservice",
    description="PyTorch Spatio-Temporal Deep Learning Service (LSTM, GRU, BiLSTM-Attention) for 15/30/60 min Traffic Speed Forecasting.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_time_index(time_index: int | None) -> int:
    data = STATE["data"]
    val_end = data["val_end"]
    max_t = len(data["timestamps"]) - max(HORIZONS) - 1
    if time_index is None:
        # Default to a lively weekday rush-hour slot inside the test set
        return min(val_end + 480, max_t)
    return max(val_end, min(time_index, max_t))


def _predict_all_sensors_at(model_key: str, t: int) -> np.ndarray:
    if model_key not in STATE["models"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not loaded. Available: {list(STATE['models'].keys())}")
    entry = STATE["models"][model_key]
    data = STATE["data"]
    mean, std = entry["mean"], entry["std"]

    t0 = t - INPUT_SEQ_LEN + 1
    speed_seq = (data["filled"][t0 : t + 1, :] - mean) / std        # (12, N)
    nbr_seq = (data["neighbor"][t0 : t + 1, :] - mean) / std        # (12, N)
    tf_seq = data["time_feats"][t0 : t + 1, :]                      # (12, 4)

    N = speed_seq.shape[1]
    # Build (N, 12, 6) batch for all 207 sensors simultaneously
    speed_b = np.transpose(speed_seq, (1, 0))[:, :, None]           # (N, 12, 1)
    nbr_b = np.transpose(nbr_seq, (1, 0))[:, :, None]               # (N, 12, 1)
    tf_b = np.broadcast_to(tf_seq[None, :, :], (N, INPUT_SEQ_LEN, 4))
    xb = torch.from_numpy(np.concatenate([speed_b, nbr_b, tf_b], axis=-1).astype(np.float32))

    with torch.no_grad():
        preds = (entry["model"](xb) * std + mean).clamp(0.0, 75.0).numpy()  # (N, 3)
    return preds


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "loadedModels": list(STATE.get("models", {}).keys()),
        "numSensors": len(STATE.get("sensor_index_map", {})),
        "testStartIndex": STATE["data"]["val_end"],
        "testEndIndex": len(STATE["data"]["timestamps"]) - max(HORIZONS) - 1,
    }


@app.get("/api/ai/sensors")
def get_sensors():
    loc = STATE["locations"]
    data = STATE["data"]
    return [
        {
            "sensorId": str(row["sensor_id"]),
            "index": int(idx),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "meanSpeedMph": round(float(np.mean(data["filled"][:, idx])), 2),
        }
        for idx, row in loc.iterrows()
    ]


@app.get("/api/ai/network-snapshot")
def get_network_snapshot(
    model: Literal["lstm", "gru", "bilstm_attention"] = Query("bilstm_attention"),
    time_index: int | None = Query(None, description="Test set time index"),
):
    """Return current + 15/30/60 min predicted speeds for all 207 sensors on the LA map."""
    t = _resolve_time_index(time_index)
    data = STATE["data"]
    loc = STATE["locations"]
    preds = _predict_all_sensors_at(model, t)  # (207, 3)

    current_speeds = data["filled"][t]
    actual_15 = data["raw"][t + HORIZONS[0]]
    actual_30 = data["raw"][t + HORIZONS[1]]
    actual_60 = data["raw"][t + HORIZONS[2]]

    sensors_out = []
    alerts = []

    for idx, row in loc.iterrows():
        cur = round(float(current_speeds[idx]), 2)
        p15 = round(float(preds[idx, 0]), 2)
        p30 = round(float(preds[idx, 1]), 2)
        p60 = round(float(preds[idx, 2]), 2)
        status_cur = classify_congestion(cur)
        status_15 = classify_congestion(p15)
        status_30 = classify_congestion(p30)
        status_60 = classify_congestion(p60)

        item = {
            "sensorId": str(row["sensor_id"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "currentSpeedMph": cur,
            "pred15MinMph": p15,
            "pred30MinMph": p30,
            "pred60MinMph": p60,
            "actual15MinMph": round(float(actual_15[idx]), 2) if actual_15[idx] > 0 else None,
            "actual30MinMph": round(float(actual_30[idx]), 2) if actual_30[idx] > 0 else None,
            "actual60MinMph": round(float(actual_60[idx]), 2) if actual_60[idx] > 0 else None,
            "statusCurrent": status_cur,
            "status15Min": status_15,
            "status30Min": status_30,
            "status60Min": status_60,
        }
        sensors_out.append(item)

        min_future = min(p15, p30, p60)
        if min_future < CONGESTION_MODERATE_MPH:
            alerts.append({
                "sensorId": item["sensorId"],
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "currentSpeedMph": cur,
                "minPredictedSpeedMph": min_future,
                "pred15MinMph": p15,
                "pred30MinMph": p30,
                "pred60MinMph": p60,
                "severity": classify_congestion(min_future),
                "speedDropMph": round(cur - min_future, 2),
            })

    alerts.sort(key=lambda a: a["minPredictedSpeedMph"])

    weather_info = None
    if STATE.get("weather") is not None:
        wrow = STATE["weather"].iloc[t]
        precip = float(wrow["precipitation_mm"])
        wcode = int(wrow["weather_code"])
        weather_info = {
            "temperatureC": round(float(wrow["temperature_c"]), 1),
            "precipitationMm": round(precip, 2),
            "windSpeedKmh": round(float(wrow["wind_speed_kmh"]), 1),
            "weatherCode": wcode,
            "conditionTr": wmo_code_to_tr(wcode),
            "isRainy": precip > 0.0,
        }

    return {
        "model": model,
        "timeIndex": t,
        "timestamp": str(data["timestamps"][t]),
        "testStartIndex": int(data["val_end"]),
        "testEndIndex": int(len(data["timestamps"]) - max(HORIZONS) - 1),
        "weather": weather_info,
        "networkSummary": {
            "avgCurrentSpeedMph": round(float(np.mean(current_speeds)), 2),
            "avgPred15MinMph": round(float(np.mean(preds[:, 0])), 2),
            "avgPred30MinMph": round(float(np.mean(preds[:, 1])), 2),
            "avgPred60MinMph": round(float(np.mean(preds[:, 2])), 2),
            "severeCount15Min": int(np.sum(preds[:, 0] < CONGESTION_SEVERE_MPH)),
            "moderateCount15Min": int(np.sum((preds[:, 0] >= CONGESTION_SEVERE_MPH) & (preds[:, 0] < CONGESTION_MODERATE_MPH))),
            "freeFlowCount15Min": int(np.sum(preds[:, 0] >= CONGESTION_MODERATE_MPH)),
        },
        "alerts": alerts[:15],
        "sensors": sensors_out,
    }


@app.get("/api/ai/predict/{sensor_id}")
def predict_sensor_detail(
    sensor_id: str,
    model: Literal["lstm", "gru", "bilstm_attention"] = Query("bilstm_attention"),
    time_index: int | None = Query(None),
    history_steps: int = Query(36, ge=12, le=144),
):
    """Detailed sensor forecast: 3-hour history + 15/30/60 min predictions across ALL loaded models."""
    idx_map = STATE["sensor_index_map"]
    if sensor_id not in idx_map:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found.")

    s_idx = idx_map[sensor_id]
    t = _resolve_time_index(time_index)
    data = STATE["data"]
    loc_row = STATE["locations"].iloc[s_idx]

    hist_start = max(0, t - history_steps + 1)
    history = [
        {
            "timestamp": str(data["timestamps"][k]),
            "speedMph": round(float(data["filled"][k, s_idx]), 2),
        }
        for k in range(hist_start, t + 1)
    ]

    model_predictions = {}
    for m_key in STATE["models"].keys():
        preds_all = _predict_all_sensors_at(m_key, t)
        p15, p30, p60 = [round(float(v), 2) for v in preds_all[s_idx]]
        model_predictions[m_key] = {
            "pred15MinMph": p15,
            "pred30MinMph": p30,
            "pred60MinMph": p60,
        }

    actual_future = {
        "actual15MinMph": round(float(data["raw"][t + HORIZONS[0], s_idx]), 2) if data["raw"][t + HORIZONS[0], s_idx] > 0 else None,
        "actual30MinMph": round(float(data["raw"][t + HORIZONS[1], s_idx]), 2) if data["raw"][t + HORIZONS[1], s_idx] > 0 else None,
        "actual60MinMph": round(float(data["raw"][t + HORIZONS[2], s_idx]), 2) if data["raw"][t + HORIZONS[2], s_idx] > 0 else None,
        "timestamps": {
            "t15": str(data["timestamps"][t + HORIZONS[0]]),
            "t30": str(data["timestamps"][t + HORIZONS[1]]),
            "t60": str(data["timestamps"][t + HORIZONS[2]]),
        },
    }

    # Extract top road-network spatial neighbors for map graph visualization (Yu et al., 2018)
    adj_row = STATE["adj_norm"][s_idx]
    nbr_indices = np.where(adj_row > 0)[0]
    nbr_indices = nbr_indices[np.argsort(-adj_row[nbr_indices])][:8]
    loc_df = STATE["locations"]
    neighbors = [
        {
            "sensorId": str(loc_df.iloc[ni]["sensor_id"]),
            "latitude": float(loc_df.iloc[ni]["latitude"]),
            "longitude": float(loc_df.iloc[ni]["longitude"]),
            "weight": round(float(adj_row[ni]), 4),
            "currentSpeedMph": round(float(data["filled"][t, ni]), 2),
        }
        for ni in nbr_indices
    ]

    return {
        "sensorId": sensor_id,
        "latitude": float(loc_row["latitude"]),
        "longitude": float(loc_row["longitude"]),
        "selectedModel": model,
        "timeIndex": t,
        "timestamp": str(data["timestamps"][t]),
        "currentSpeedMph": round(float(data["filled"][t, s_idx]), 2),
        "neighborAvgSpeedMph": round(float(data["neighbor"][t, s_idx]), 2),
        "neighbors": neighbors,
        "history": history,
        "predictions": model_predictions,
        "actualFuture": actual_future,
    }


@app.post("/api/ai/predict/custom")
def predict_custom(req: CustomPredictRequest):
    """Real-time inference endpoint accepting 12 external 5-minute speed readings."""
    if len(req.speeds) != INPUT_SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {INPUT_SEQ_LEN} historical speed values (last 60 min, 5-min steps), got {len(req.speeds)}.",
        )
    if req.model not in STATE["models"]:
        raise HTTPException(status_code=404, detail=f"Model '{req.model}' not loaded.")

    speeds = np.array(req.speeds, dtype=np.float32)
    nbr_speeds = (
        np.array(req.neighborSpeeds, dtype=np.float32)
        if req.neighborSpeeds and len(req.neighborSpeeds) == INPUT_SEQ_LEN
        else speeds.copy()
    )

    # Build cyclical time features for the 12 historical 5-min steps ending at (hour, minute)
    end_min = (req.hour % 24) * 60 + (req.minute % 60)
    t_feats = []
    for step in range(-INPUT_SEQ_LEN + 1, 1):
        m_total = (end_min + step * 5) % (24 * 60)
        tod = m_total / (24.0 * 60.0)
        dow = (req.dayOfWeek % 7) / 7.0
        t_feats.append([
            np.sin(2.0 * np.pi * tod),
            np.cos(2.0 * np.pi * tod),
            np.sin(2.0 * np.pi * dow),
            np.cos(2.0 * np.pi * dow),
        ])
    tf_arr = np.array(t_feats, dtype=np.float32)  # (12, 4)

    all_preds = {}
    for m_key, entry in STATE["models"].items():
        mean, std = entry["mean"], entry["std"]
        s_norm = ((speeds - mean) / std)[:, None]
        n_norm = ((nbr_speeds - mean) / std)[:, None]
        xb = torch.from_numpy(np.concatenate([s_norm, n_norm, tf_arr], axis=-1)[None, :, :].astype(np.float32))
        with torch.no_grad():
            p = (entry["model"](xb) * std + mean).clamp(0.0, 75.0).numpy()[0]
        p15, p30, p60 = [round(float(v), 2) for v in p]
        all_preds[m_key] = {
            "pred15MinMph": p15,
            "pred30MinMph": p30,
            "pred60MinMph": p60,
            "status15Min": classify_congestion(p15),
            "status30Min": classify_congestion(p30),
            "status60Min": classify_congestion(p60),
        }

    return {
        "selectedModel": req.model,
        "inputCurrentSpeedMph": round(float(speeds[-1]), 2),
        "primaryPrediction": all_preds[req.model],
        "allModels": all_preds,
    }


@app.get("/api/ai/experiments")
def get_experiments():
    """Return full academic model comparison & ablation study results."""
    exp_path = RESULTS_DIR / "experiment_results.json"
    if exp_path.exists():
        return json.loads(exp_path.read_text(encoding="utf-8"))
    return STATE.get("experiments", {})
