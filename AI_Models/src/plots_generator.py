"""
Publication-grade (300 DPI) academic plots for the METR-LA Traffic Flow Forecasting thesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PALETTE = {
    "Persistence": "#94a3b8",
    "Historical Avg": "#64748b",
    "LSTM": "#2563eb",
    "GRU": "#059669",
    "BiLSTM-Attention": "#7c3aed",
}


def plot_eda_diurnal_weekly(df: pd.DataFrame) -> str:
    """Plot 1: Diurnal rush-hour & weekday vs weekend traffic speed profile."""
    df_nz = df.replace(0.0, np.nan)
    mean_speed = df_nz.mean(axis=1)
    eda = pd.DataFrame({
        "speed": mean_speed,
        "hour": df.index.hour + df.index.minute / 60.0,
        "is_weekend": df.index.dayofweek >= 5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)

    # Left: Weekday vs Weekend diurnal curve
    wd = eda[~eda["is_weekend"]].groupby("hour")["speed"].agg(["mean", "std"])
    we = eda[eda["is_weekend"]].groupby("hour")["speed"].agg(["mean", "std"])

    axes[0].plot(wd.index, wd["mean"], color="#2563eb", lw=2.2, label="Hafta İçi (Weekday)")
    axes[0].fill_between(wd.index, wd["mean"] - wd["std"], wd["mean"] + wd["std"], color="#2563eb", alpha=0.14)
    axes[0].plot(we.index, we["mean"], color="#f59e0b", lw=2.2, linestyle="--", label="Hafta Sonu (Weekend)")
    axes[0].fill_between(we.index, we["mean"] - we["std"], we["mean"] + we["std"], color="#f59e0b", alpha=0.14)
    axes[0].axvspan(7, 9.5, color="#ef4444", alpha=0.08, label="Sabah Zirve (07:00-09:30)")
    axes[0].axvspan(16, 19, color="#dc2626", alpha=0.08, label="Akşam Zirve (16:00-19:00)")
    axes[0].set_title("METR-LA Günlük Ortalama Trafik Hızı Profili (207 Sensör)", fontweight="bold")
    axes[0].set_xlabel("Günün Saati")
    axes[0].set_ylabel("Ortalama Hız (mph)")
    axes[0].set_xlim(0, 24)
    axes[0].set_xticks(range(0, 25, 3))
    axes[0].legend(loc="lower left", fontsize=8.5)

    # Right: Speed distribution histogram
    vals = df.values[df.values > 0]
    axes[1].hist(vals, bins=55, color="#0ea5e9", edgecolor="white", alpha=0.85, density=True)
    axes[1].axvline(np.mean(vals), color="#ef4444", linestyle="--", lw=1.8, label=f"Ortalama: {np.mean(vals):.1f} mph")
    axes[1].axvline(np.median(vals), color="#10b981", linestyle=":", lw=1.8, label=f"Medyan: {np.median(vals):.1f} mph")
    axes[1].set_title("METR-LA Hız Dağılımı (Sıfır Hariç Gerçek Ölçümler)", fontweight="bold")
    axes[1].set_xlabel("Hız (mph)")
    axes[1].set_ylabel("Yoğunluk")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "01_eda_daily_weekly_pattern.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_training_curves(histories: dict[str, dict]) -> str:
    """Plot 2: Train & validation masked MAE loss curves for main deep learning models."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=300)

    for name, hist in histories.items():
        color = PALETTE.get(name, "#334155")
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], marker="o", ms=4, lw=2, color=color, label=name)
        axes[1].plot(epochs, hist["val_loss"], marker="s", ms=4, lw=2, color=color, label=name)

    axes[0].set_title("Eğitim Kaybı (Training Masked MAE)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked MAE (mph)")
    axes[0].legend()

    axes[1].set_title("Doğrulama Kaybı (Validation Masked MAE)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Masked MAE (mph)")
    axes[1].legend()

    plt.tight_layout()
    out = PLOTS_DIR / "02_training_validation_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_model_comparison_horizons(model_results: dict[str, dict]) -> str:
    """Plot 3: Grouped bar chart of MAE and RMSE across 15, 30, 60 min horizons."""
    horizons = ["15 min", "30 min", "60 min"]
    models = list(model_results.keys())
    x = np.arange(len(horizons))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=300)

    for m_idx, model_name in enumerate(models):
        color = PALETTE.get(model_name, "#334155")
        maes = [model_results[model_name]["horizons"][h]["mae"] for h in horizons]
        rmses = [model_results[model_name]["horizons"][h]["rmse"] for h in horizons]
        offset = (m_idx - (len(models) - 1) / 2) * width

        b1 = axes[0].bar(x + offset, maes, width, label=model_name, color=color, alpha=0.9)
        b2 = axes[1].bar(x + offset, rmses, width, label=model_name, color=color, alpha=0.9)

        for bar in b1:
            h_val = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h_val + 0.08, f"{h_val:.2f}", ha="center", va="bottom", fontsize=7.5)
        for bar in b2:
            h_val = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h_val + 0.12, f"{h_val:.2f}", ha="center", va="bottom", fontsize=7.5)

    axes[0].set_title("Tahmin Ufkuna Göre MAE Karşılaştırması (Düşük = Daha İyi)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["15 Dakika", "30 Dakika", "60 Dakika"])
    axes[0].set_ylabel("Masked MAE (mph)")
    axes[0].legend(fontsize=8.5)

    axes[1].set_title("Tahmin Ufkuna Göre RMSE Karşılaştırması (Düşük = Daha İyi)", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["15 Dakika", "30 Dakika", "60 Dakika"])
    axes[1].set_ylabel("Masked RMSE (mph)")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    out = PLOTS_DIR / "03_model_comparison_horizons.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_ablation_study(ablation_results: list[dict]) -> str:
    """Plot 4: Hyperparameter & spatial feature ablation comparison."""
    labels = [r["experiment"] for r in ablation_results]
    maes = [r["metrics"]["overall"]["mae"] for r in ablation_results]
    rmses = [r["metrics"]["overall"]["rmse"] for r in ablation_results]

    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=300)

    bars1 = axes[0].barh(y, maes, color="#3b82f6", alpha=0.88)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Genel Ortalama MAE (mph)")
    axes[0].set_title("Hiperparametre & Uzamsal Özellik Ablasyonu (MAE)", fontweight="bold")
    for bar, v in zip(bars1, maes):
        axes[0].text(v + 0.05, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)

    bars2 = axes[1].barh(y, rmses, color="#8b5cf6", alpha=0.88)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([""] * len(labels))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Genel Ortalama RMSE (mph)")
    axes[1].set_title("Hiperparametre & Uzamsal Özellik Ablasyonu (RMSE)", fontweight="bold")
    for bar, v in zip(bars2, rmses):
        axes[1].text(v + 0.08, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)

    plt.tight_layout()
    out = PLOTS_DIR / "04_ablation_hyperparameters.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_actual_vs_predicted_timeseries(
    timestamps: pd.DatetimeIndex,
    actual_15: np.ndarray,
    preds_by_model: dict[str, np.ndarray],
    sensor_id: str,
) -> str:
    """Plot 5: 48-hour real vs predicted time series on a representative LA highway sensor."""
    fig, ax = plt.subplots(figsize=(13.5, 4.8), dpi=300)
    mask = actual_15 > 0
    ax.plot(timestamps[mask], actual_15[mask], color="#0f172a", lw=2.0, label="Gerçek Hız (Actual)", zorder=5)

    for model_name, pred in preds_by_model.items():
        ax.plot(
            timestamps[mask],
            pred[mask],
            lw=1.6,
            alpha=0.88,
            color=PALETTE.get(model_name, "#334155"),
            linestyle="--" if model_name != "BiLSTM-Attention" else "-",
            label=f"{model_name} (+15 dk Tahmin)",
        )

    ax.set_title(f"Gerçek vs. Tahmin Edilen Trafik Hızı — Sensör #{sensor_id} (Test Seti, 48 Saat)", fontweight="bold")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Hız (mph)")
    ax.legend(loc="lower left")
    fig.autofmt_xdate()

    plt.tight_layout()
    out = PLOTS_DIR / "05_actual_vs_predicted_timeseries.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_error_distribution_scatter(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, r2_val: float) -> str:
    """Plot 6: Residual histogram and actual-vs-predicted scatter plot for the best model."""
    mask = y_true > 0
    yt = y_true[mask]
    yp = y_pred[mask]
    residuals = yp - yt

    # Subsample scatter points for crisp visualization
    rng = np.random.default_rng(42)
    idx = rng.choice(len(yt), size=min(6000, len(yt)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=300)

    axes[0].scatter(yt[idx], yp[idx], s=8, alpha=0.25, color="#2563eb", edgecolors="none")
    axes[0].plot([0, 72], [0, 72], color="#ef4444", linestyle="--", lw=1.8, label="İdeal Doğru (y = x)")
    axes[0].set_title(f"{model_name} — Gerçek vs. Tahmin (+15 dk, R² = {r2_val:.4f})", fontweight="bold")
    axes[0].set_xlabel("Gerçek Hız (mph)")
    axes[0].set_ylabel("Tahmin Edilen Hız (mph)")
    axes[0].set_xlim(0, 72)
    axes[0].set_ylim(0, 72)
    axes[0].legend()

    clipped_res = residuals[(residuals > -25) & (residuals < 25)]
    axes[1].hist(clipped_res, bins=60, color="#7c3aed", alpha=0.82, edgecolor="white", density=True)
    axes[1].axvline(0.0, color="#0f172a", linestyle="--", lw=1.6, label=f"Sıfır Hata (Ort: {np.mean(residuals):.2f})")
    axes[1].set_title(f"{model_name} — Tahmin Hata Dağılımı (Residual)", fontweight="bold")
    axes[1].set_xlabel("Hata (Tahmin - Gerçek, mph)")
    axes[1].set_ylabel("Yoğunluk")
    axes[1].legend()

    plt.tight_layout()
    out = PLOTS_DIR / "06_error_distribution_scatter.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)
