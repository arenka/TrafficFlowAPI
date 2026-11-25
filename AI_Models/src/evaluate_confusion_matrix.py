"""
Post-evaluation script: Converts continuous traffic speed predictions (15, 30, 60 min)
for ALL trained deep learning models (LSTM, GRU, BiLSTM-Attention) into 3 operational
Decision Support congestion states and generates:
  - AI_Models/plots/07_congestion_confusion_matrix.png (300 DPI for best model)
  - AI_Models/results/congestion_classification_report.csv (all models x all horizons)
  - Updates AI_Models/results/experiment_results.json with full interactive 3x3 confusion matrices
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    CONGESTION_MODERATE_MPH,
    CONGESTION_SEVERE_MPH,
    MODELS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
)
from data_loader import (
    build_spatial_weight_matrix,
    build_window_tensors,
    load_metr_la_dataframe,
    prepare_spatiotemporal_arrays,
)
from models import build_model

CLASS_LABELS = ["Kritik (<25)", "Orta (25-45)", "Akıcı (>=45)"]
HORIZON_NAMES = ["15 Dakika", "30 Dakika", "60 Dakika"]


def speed_to_class(arr: np.ndarray) -> np.ndarray:
    """0 = Severe (<25), 1 = Moderate (25..45), 2 = FreeFlow (>=45)."""
    cls = np.full(arr.shape, 2, dtype=np.int32)
    cls[arr < CONGESTION_MODERATE_MPH] = 1
    cls[arr < CONGESTION_SEVERE_MPH] = 0
    return cls


def evaluate_checkpoint(ckpt_path, X_test, Y_test):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=2048, shuffle=False)
    mean, std = float(ckpt["scaler"]["mean"]), float(ckpt["scaler"]["std"])
    preds_all, trues_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = (model(xb) * std + mean).clamp(0.0, 75.0).numpy()
            preds_all.append(p)
            trues_all.append(yb.numpy())
    return np.concatenate(trues_all, axis=0), np.concatenate(preds_all, axis=0)


def main():
    exp_path = RESULTS_DIR / "experiment_results.json"
    exp_json = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.exists() else {}
    best_model_name = exp_json.get("best_model", "BiLSTM-Attention")

    df = load_metr_la_dataframe()
    adj = build_spatial_weight_matrix(list(df.columns))
    data = prepare_spatiotemporal_arrays(df, adj)

    X_test, Y_test = build_window_tensors(
        data["filled"], data["neighbor"], data["raw"], data["time_feats"],
        data["mean"], data["std"], data["val_end"], len(data["timestamps"]),
        stride=12, use_spatial=True,
    )

    ckpt_map = {
        "LSTM": MODELS_DIR / "lstm_best.pth",
        "GRU": MODELS_DIR / "gru_best.pth",
        "BiLSTM-Attention": MODELS_DIR / "bilstm_attention_best.pth",
    }

    report_rows = []
    all_cm_data = {}
    best_plot_payload = None

    for model_name, ckpt_path in ckpt_map.items():
        if not ckpt_path.exists():
            continue
        y_true, y_pred = evaluate_checkpoint(ckpt_path, X_test, Y_test)
        all_cm_data[model_name] = {}

        horizon_cms = []
        for h_idx, h_label in enumerate(HORIZON_NAMES):
            yt = y_true[:, h_idx]
            yp = y_pred[:, h_idx]
            mask = yt > 0.0
            c_true = speed_to_class(yt[mask])
            c_pred = speed_to_class(yp[mask])

            cm = confusion_matrix(c_true, c_pred, labels=[0, 1, 2])
            cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
            acc = float((c_true == c_pred).mean() * 100.0)

            rep = classification_report(
                c_true, c_pred, labels=[0, 1, 2], target_names=CLASS_LABELS, output_dict=True, zero_division=0
            )

            all_cm_data[model_name][h_label] = {
                "labels": CLASS_LABELS,
                "matrix": cm.tolist(),
                "matrix_pct": np.round(cm_norm, 2).tolist(),
                "accuracy_pct": round(acc, 2),
                "macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
                "weighted_f1": round(float(rep["weighted avg"]["f1-score"]), 4),
                "severe_recall": round(float(rep[CLASS_LABELS[0]]["recall"]), 4),
                "severe_precision": round(float(rep[CLASS_LABELS[0]]["precision"]), 4),
            }
            horizon_cms.append((h_label, cm, cm_norm, acc))

            for cls_name in CLASS_LABELS + ["macro avg", "weighted avg"]:
                d = rep[cls_name]
                report_rows.append({
                    "Model": model_name,
                    "Horizon": h_label,
                    "Class": cls_name,
                    "Precision": round(float(d["precision"]), 4),
                    "Recall": round(float(d["recall"]), 4),
                    "F1-Score": round(float(d["f1-score"]), 4),
                    "Support": int(d["support"]),
                    "Accuracy (%)": round(acc, 2),
                })

        if model_name == best_model_name or best_plot_payload is None:
            best_plot_payload = (model_name, horizon_cms)

    # Generate 300 DPI thesis figure for the best model
    if best_plot_payload is not None:
        plot_model_name, horizon_cms = best_plot_payload
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)
        for h_idx, (h_label, cm, cm_norm, acc) in enumerate(horizon_cms):
            annot = np.empty_like(cm, dtype=object)
            for i in range(3):
                for j in range(3):
                    annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

            sns.heatmap(
                cm_norm,
                annot=annot,
                fmt="",
                cmap="Blues",
                vmin=0,
                vmax=100,
                cbar=(h_idx == 2),
                xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS,
                ax=axes[h_idx],
                annot_kws={"size": 8.5},
            )
            axes[h_idx].set_title(f"{plot_model_name} — {h_label}\n(Doğruluk: %{acc:.2f})", fontweight="bold")
            axes[h_idx].set_xlabel("Tahmin Edilen Durum")
            if h_idx == 0:
                axes[h_idx].set_ylabel("Gerçek Trafik Durumu")

        plt.suptitle(
            f"Karar Destek Sistemi Trafik Yoğunluk Sınıflandırma Karmaşıklık Matrisi ({plot_model_name})",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        out_plot = PLOTS_DIR / "07_congestion_confusion_matrix.png"
        plt.savefig(out_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_plot}")

    out_csv = RESULTS_DIR / "congestion_classification_report.csv"
    pd.DataFrame(report_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    exp_json["confusion_matrices"] = all_cm_data
    exp_path.write_text(json.dumps(exp_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated: {exp_path} with interactive confusion matrices for {list(all_cm_data.keys())}")


if __name__ == "__main__":
    main()
