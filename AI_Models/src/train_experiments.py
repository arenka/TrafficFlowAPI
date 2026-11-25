"""
End-to-End Academic Experiment Runner for METR-LA Dynamic Traffic Flow Forecasting.
Trains & evaluates:
  - Baselines: Persistence (Last Value), Historical Average (HA)
  - Main Models: LSTM, GRU, BiLSTM-Attention (15, 30, 60 min multi-horizon)
  - Ablation Study: Layer depth, hidden units, and spatial neighbor feature contribution

Saves model weights, scaler metadata, CSV/JSON tables, and 300-DPI thesis plots.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    HORIZONS,
    INPUT_SEQ_LEN,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)
from data_loader import (
    build_spatial_weight_matrix,
    build_window_tensors,
    load_metr_la_dataframe,
    prepare_spatiotemporal_arrays,
)
from metrics import evaluate_multi_horizon
from models import build_model
from plots_generator import (
    plot_ablation_study,
    plot_actual_vs_predicted_timeseries,
    plot_eda_diurnal_weekly,
    plot_error_distribution_scatter,
    plot_model_comparison_horizons,
    plot_training_curves,
)


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = (target > 0.0).float()
    diff = torch.abs(pred - target) * mask
    return diff.sum() / mask.sum().clamp(min=1.0)


def evaluate_baselines(data: dict, test_stride: int = 12) -> dict[str, dict]:
    """Compute Persistence and Historical Average baselines on the exact same test windows."""
    raw = data["raw"]
    filled = data["filled"]
    timestamps = data["timestamps"]
    train_end = data["train_end"]
    val_end = data["val_end"]
    T, N = raw.shape

    # Build Historical Average lookup table (day_of_week, slot_of_day) -> (N,) from TRAIN split only
    train_df = pd.DataFrame(raw[:train_end], index=timestamps[:train_end]).replace(0.0, np.nan)
    slot_key = train_df.index.day_of_week * 288 + (train_df.index.hour * 12 + train_df.index.minute // 5)
    ha_table = train_df.groupby(slot_key).mean()
    global_sensor_mean = train_df.mean(axis=0).fillna(data["mean"]).values.astype(np.float32)

    max_h = max(HORIZONS)
    test_t = np.arange(max(val_end, INPUT_SEQ_LEN - 1), T - max_h, test_stride, dtype=np.int32)

    y_true_list, persist_list, ha_list = [], [], []
    for t in test_t:
        yt = np.stack([raw[t + h] for h in HORIZONS], axis=-1)              # (N, 3)
        yp_persist = np.repeat(filled[t : t + 1].T, len(HORIZONS), axis=-1) # (N, 3)

        ha_cols = []
        for h in HORIZONS:
            ts = timestamps[t + h]
            k = ts.day_of_week * 288 + (ts.hour * 12 + ts.minute // 5)
            if k in ha_table.index:
                vals = ha_table.loc[k].values.astype(np.float32)
                vals = np.where(np.isnan(vals), global_sensor_mean, vals)
            else:
                vals = global_sensor_mean
            ha_cols.append(vals)
        yp_ha = np.stack(ha_cols, axis=-1)

        y_true_list.append(yt)
        persist_list.append(yp_persist)
        ha_list.append(yp_ha)

    y_true = np.concatenate(y_true_list, axis=0)
    y_persist = np.concatenate(persist_list, axis=0)
    y_ha = np.concatenate(ha_list, axis=0)

    return {
        "Persistence": evaluate_multi_horizon(y_true, y_persist),
        "Historical Avg": evaluate_multi_horizon(y_true, y_ha),
    }


def train_and_evaluate_model(
    model_type: str,
    data: dict,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 3e-3,
    epochs: int = 12,
    batch_size: int = 512,
    use_spatial: bool = True,
    train_stride: int = 24,
    val_stride: int = 24,
    test_stride: int = 12,
    save_name: str | None = None,
) -> dict:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 6 if use_spatial else 5
    model = build_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_horizons=len(HORIZONS),
    ).to(device)

    train_ds = TensorDataset(*build_window_tensors(
        data["filled"], data["neighbor"], data["raw"], data["time_feats"],
        data["mean"], data["std"], 0, data["train_end"], stride=train_stride, use_spatial=use_spatial,
    ))
    val_ds = TensorDataset(*build_window_tensors(
        data["filled"], data["neighbor"], data["raw"], data["time_feats"],
        data["mean"], data["std"], data["train_end"], data["val_end"], stride=val_stride, use_spatial=use_spatial,
    ))
    test_ds = TensorDataset(*build_window_tensors(
        data["filled"], data["neighbor"], data["raw"], data["time_feats"],
        data["mean"], data["std"], data["val_end"], len(data["timestamps"]), stride=test_stride, use_spatial=use_spatial,
    ))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    mean_t = torch.tensor(data["mean"], dtype=torch.float32, device=device)
    std_t = torch.tensor(data["std"], dtype=torch.float32, device=device)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience = 4
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_sum, tr_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred_norm = model(xb)
            pred_mph = pred_norm * std_t + mean_t
            loss = masked_mae_loss(pred_mph, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss_sum += float(loss.item())
            tr_batches += 1

        scheduler.step()
        train_mae = tr_loss_sum / max(tr_batches, 1)

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred_mph = model(xb) * std_t + mean_t
                val_loss_sum += float(masked_mae_loss(pred_mph, yb).item())
                val_batches += 1
        val_mae = val_loss_sum / max(val_batches, 1)

        history["train_loss"].append(round(train_mae, 4))
        history["val_loss"].append(round(val_mae, 4))
        print(f"  [{save_name or model_type}] Epoch {epoch:02d}/{epochs} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}", flush=True)

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch}.", flush=True)
                break

    elapsed = round(time.time() - t0, 2)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred_mph = (model(xb) * std_t + mean_t).clamp(0.0, 75.0).cpu().numpy()
            preds_all.append(pred_mph)
            trues_all.append(yb.numpy())

    y_pred = np.concatenate(preds_all, axis=0)
    y_true = np.concatenate(trues_all, axis=0)
    metrics = evaluate_multi_horizon(y_true, y_pred)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if save_name:
        ckpt_path = MODELS_DIR / f"{save_name}_best.pth"
        torch.save(
            {
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "config": {
                    "input_dim": input_dim,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "num_horizons": len(HORIZONS),
                    "use_spatial": use_spatial,
                },
                "scaler": {"mean": data["mean"], "std": data["std"]},
                "metrics": metrics,
            },
            ckpt_path,
        )

    return {
        "model": model,
        "metrics": metrics,
        "history": history,
        "train_time_sec": elapsed,
        "num_params": num_params,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def predict_sensor_slice(model: nn.Module, data: dict, sensor_idx: int, start_t: int, steps: int = 576) -> tuple[np.ndarray, np.ndarray]:
    """Generate consecutive +15 min predictions over a 48-hour (576 x 5min) test window for plotting."""
    device = next(model.parameters()).device
    model.eval()
    filled_norm = (data["filled"] - data["mean"]) / data["std"]
    nbr_norm = (data["neighbor"] - data["mean"]) / data["std"]

    xs = []
    actuals = []
    for t in range(start_t, start_t + steps):
        t0 = t - INPUT_SEQ_LEN + 1
        s_seq = filled_norm[t0 : t + 1, sensor_idx : sensor_idx + 1]
        n_seq = nbr_norm[t0 : t + 1, sensor_idx : sensor_idx + 1]
        tf_seq = data["time_feats"][t0 : t + 1, :]
        xs.append(np.concatenate([s_seq, n_seq, tf_seq], axis=-1))
        actuals.append(data["raw"][t + HORIZONS[0], sensor_idx])

    xb = torch.tensor(np.array(xs, dtype=np.float32), device=device)
    with torch.no_grad():
        preds = (model(xb) * data["std"] + data["mean"]).clamp(0.0, 75.0).cpu().numpy()[:, 0]
    return np.array(actuals, dtype=np.float32), preds


def main():
    print("=== METR-LA Dynamic Traffic Flow Forecasting — Academic Pipeline ===", flush=True)
    df = load_metr_la_dataframe()
    sensor_ids = list(df.columns)
    adj_norm = build_spatial_weight_matrix(sensor_ids)
    data = prepare_spatiotemporal_arrays(df, adj_norm)
    print(f"Dataset loaded: {df.shape} | Train mean={data['mean']:.2f} mph, std={data['std']:.2f} mph", flush=True)

    # 1. EDA Plot
    plot_eda_diurnal_weekly(df)
    print("Generated 01_eda_daily_weekly_pattern.png", flush=True)

    # 2. Baselines
    print("\n--- Evaluating Baselines ---", flush=True)
    baseline_results = evaluate_baselines(data, test_stride=12)
    for b_name, b_m in baseline_results.items():
        print(f"  {b_name:16s} | Overall MAE={b_m['overall']['mae']:.3f}, RMSE={b_m['overall']['rmse']:.3f}, R2={b_m['overall']['r2']:.4f}", flush=True)

    # 3. Main Deep Learning Models
    print("\n--- Training Main Deep Learning Models (15/30/60 min) ---", flush=True)
    main_configs = [
        ("LSTM", "lstm", "lstm", dict(hidden_size=64, num_layers=2, dropout=0.2, lr=4e-3, epochs=8, batch_size=1024, train_stride=36)),
        ("GRU", "gru", "gru", dict(hidden_size=64, num_layers=2, dropout=0.2, lr=4e-3, epochs=8, batch_size=1024, train_stride=36)),
        ("BiLSTM-Attention", "bilstm_attention", "bilstm_attention", dict(hidden_size=64, num_layers=2, dropout=0.2, lr=3e-3, epochs=8, batch_size=1024, train_stride=36)),
    ]

    all_model_metrics = dict(baseline_results)
    histories = {}
    trained_runs = {}

    for display_name, m_type, save_name, kwargs in main_configs:
        print(f"\nTraining {display_name}...", flush=True)
        run = train_and_evaluate_model(m_type, data, save_name=save_name, **kwargs)
        all_model_metrics[display_name] = run["metrics"]
        histories[display_name] = run["history"]
        trained_runs[display_name] = run
        ov = run["metrics"]["overall"]
        print(f"  => {display_name} Test Overall: MAE={ov['mae']:.4f}, RMSE={ov['rmse']:.4f}, MAPE={ov['mape']:.2f}%, R2={ov['r2']:.4f} ({run['train_time_sec']}s)", flush=True)

    # 4. Hyperparameter & Spatial Feature Ablation Study
    print("\n--- Running Hyperparameter & Spatial-Feature Ablation Study ---", flush=True)
    ablation_configs = [
        ("LSTM (1 Katman, h=32)", "lstm", dict(hidden_size=32, num_layers=1, dropout=0.1, lr=4e-3, epochs=6, batch_size=1024, train_stride=36, use_spatial=True)),
        ("LSTM (2 Katman, h=64, Uzamsal Yok)", "lstm", dict(hidden_size=64, num_layers=2, dropout=0.2, lr=4e-3, epochs=6, batch_size=1024, train_stride=36, use_spatial=False)),
        ("LSTM (2 Katman, h=64, Uzamsal Var)", None, None),  # Reuse main LSTM
        ("LSTM (3 Katman, h=96, Uzamsal Var)", "lstm", dict(hidden_size=96, num_layers=3, dropout=0.25, lr=3e-3, epochs=6, batch_size=1024, train_stride=36, use_spatial=True)),
    ]

    ablation_results = []
    for exp_name, m_type, kwargs in ablation_configs:
        if m_type is None:
            ref = trained_runs["LSTM"]
            ablation_results.append({
                "experiment": exp_name,
                "num_layers": 2,
                "hidden_size": 64,
                "use_spatial": True,
                "num_params": ref["num_params"],
                "train_time_sec": ref["train_time_sec"],
                "metrics": ref["metrics"],
            })
        else:
            print(f"\nAblation: {exp_name}...", flush=True)
            run = train_and_evaluate_model(m_type, data, save_name=None, **kwargs)
            ablation_results.append({
                "experiment": exp_name,
                "num_layers": kwargs["num_layers"],
                "hidden_size": kwargs["hidden_size"],
                "use_spatial": kwargs["use_spatial"],
                "num_params": run["num_params"],
                "train_time_sec": run["train_time_sec"],
                "metrics": run["metrics"],
            })

    # 5. Generate remaining thesis plots
    plot_training_curves(histories)
    plot_model_comparison_horizons(all_model_metrics)
    plot_ablation_study(ablation_results)

    # Pick a high-variance busy sensor in the test set for the 48-hour visual
    test_start = data["val_end"] + 200
    sensor_stds = np.std(data["filled"][test_start : test_start + 576], axis=0)
    rep_sensor_idx = int(np.argmax(sensor_stds))
    rep_sensor_id = sensor_ids[rep_sensor_idx]
    preds_48h = {}
    actual_48h = None
    for display_name in ["LSTM", "GRU", "BiLSTM-Attention"]:
        act, prd = predict_sensor_slice(trained_runs[display_name]["model"], data, rep_sensor_idx, test_start, steps=576)
        actual_48h = act
        preds_48h[display_name] = prd
    ts_slice = data["timestamps"][test_start + HORIZONS[0] : test_start + HORIZONS[0] + 576]
    plot_actual_vs_predicted_timeseries(ts_slice, actual_48h, preds_48h, rep_sensor_id)

    # Best DL model scatter & residual plot
    best_dl_name = min(["LSTM", "GRU", "BiLSTM-Attention"], key=lambda k: all_model_metrics[k]["overall"]["mae"])
    best_run = trained_runs[best_dl_name]
    plot_error_distribution_scatter(
        best_run["y_true"][:, 0],
        best_run["y_pred"][:, 0],
        best_dl_name,
        best_run["metrics"]["horizons"]["15 min"]["r2"],
    )

    # 6. Export structured JSON & CSV summaries for Thesis & Backend API
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "METR-LA (Los Angeles Highway Loop Detectors)",
            "num_sensors": len(sensor_ids),
            "num_timesteps": int(df.shape[0]),
            "interval_minutes": 5,
            "date_start": str(df.index.min()),
            "date_end": str(df.index.max()),
            "missing_zero_pct": round(float((data["raw"] == 0.0).mean() * 100.0), 2),
            "train_mean_mph": round(data["mean"], 4),
            "train_std_mph": round(data["std"], 4),
            "split": {"train": 0.70, "val": 0.10, "test": 0.20},
        },
        "best_model": best_dl_name,
        "models": {
            name: {
                "metrics": all_model_metrics[name],
                "num_params": trained_runs[name]["num_params"] if name in trained_runs else 0,
                "train_time_sec": trained_runs[name]["train_time_sec"] if name in trained_runs else 0.0,
                "history": histories.get(name, None),
            }
            for name in all_model_metrics
        },
        "ablation_study": ablation_results,
    }

    with open(RESULTS_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Export CSV tables for direct inclusion in thesis document
    rows = []
    for name, m in all_model_metrics.items():
        for h in ["15 min", "30 min", "60 min"]:
            rows.append({
                "Model": name,
                "Horizon": h,
                "MAE (mph)": m["horizons"][h]["mae"],
                "RMSE (mph)": m["horizons"][h]["rmse"],
                "MAPE (%)": m["horizons"][h]["mape"],
                "R2": m["horizons"][h]["r2"],
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    ab_rows = [
        {
            "Experiment": r["experiment"],
            "Layers": r["num_layers"],
            "Hidden Size": r["hidden_size"],
            "Spatial Neighbor": r["use_spatial"],
            "Params": r["num_params"],
            "Overall MAE": r["metrics"]["overall"]["mae"],
            "Overall RMSE": r["metrics"]["overall"]["rmse"],
            "Overall MAPE (%)": r["metrics"]["overall"]["mape"],
            "Overall R2": r["metrics"]["overall"]["r2"],
        }
        for r in ablation_results
    ]
    pd.DataFrame(ab_rows).to_csv(RESULTS_DIR / "ablation_study.csv", index=False)

    print(f"\n=== ALL EXPERIMENTS COMPLETE! Best Model: {best_dl_name} ===", flush=True)
    print(f"Results saved to: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
