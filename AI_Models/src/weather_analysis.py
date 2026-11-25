"""
Real Meteorological Impact Analysis on METR-LA Traffic Flow (2012-03-01 to 2012-06-27).
Uses real historical Los Angeles hourly weather from Open-Meteo Archive API aligned
to the 5-minute METR-LA sensor timestamps.

Outputs:
  - AI_Models/plots/08_weather_traffic_impact.png (300 DPI)
  - AI_Models/results/weather_impact.csv
  - Updates AI_Models/results/experiment_results.json with weather_impact summary
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import CONGESTION_SEVERE_MPH, DATA_DIR, PLOTS_DIR, RESULTS_DIR
from data_loader import load_metr_la_dataframe

WEATHER_PATH = DATA_DIR / "la_weather_2012.csv"


def wmo_code_to_tr(code: int) -> str:
    if code == 0:
        return "Açık / Güneşli"
    if code in (1, 2):
        return "Parçalı Bulutlu"
    if code == 3:
        return "Kapalı / Bulutlu"
    if code in (45, 48):
        return "Sisli"
    if code in (51, 53, 55):
        return "Çiseleyen Yağmur"
    if code in (61, 63, 65, 80, 81, 82):
        return "Yağmurlu"
    return "Diğer"


def align_weather_to_metr_la(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    wdf = pd.read_csv(WEATHER_PATH)
    wdf["timestamp"] = pd.to_datetime(wdf["timestamp"])
    wdf = wdf.set_index("timestamp").sort_index()
    # Align each 5-min METR-LA timestamp to its corresponding hourly weather observation
    hourly_keys = timestamps.floor("h")
    aligned = wdf.reindex(hourly_keys).ffill().bfill()
    aligned.index = timestamps
    return aligned


def main():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
    df = load_metr_la_dataframe()
    weather = align_weather_to_metr_la(df.index)

    # Mask zero sensor dropouts
    df_nz = df.replace(0.0, np.nan)
    net_mean_speed = df_nz.mean(axis=1)
    severe_pct = (df_nz < CONGESTION_SEVERE_MPH).mean(axis=1) * 100.0

    combined = pd.DataFrame({
        "speed_mph": net_mean_speed,
        "severe_pct": severe_pct,
        "hour": df.index.hour + df.index.minute / 60.0,
        "is_rush": df.index.hour.isin([7, 8, 9, 16, 17, 18]),
        "temp_c": weather["temperature_c"].values,
        "precip_mm": weather["precipitation_mm"].values,
        "wind_kmh": weather["wind_speed_kmh"].values,
        "wmo": weather["weather_code"].values.astype(int),
    })

    def bucket(p: float) -> str:
        if p <= 0.0:
            return "Kuru / Yağışsız (0 mm)"
        if p <= 1.0:
            return "Hafif Yağmur (0–1 mm/sa)"
        return "Orta/Şiddetli Yağmur (>1 mm/sa)"

    combined["condition"] = combined["precip_mm"].apply(bucket)
    order = ["Kuru / Yağışsız (0 mm)", "Hafif Yağmur (0–1 mm/sa)", "Orta/Şiddetli Yağmur (>1 mm/sa)"]

    rows = []
    for cond in order:
        sub = combined[combined["condition"] == cond]
        rush_sub = sub[sub["is_rush"]]
        rows.append({
            "Weather Condition": cond,
            "5-Min Intervals": int(len(sub)),
            "Avg Speed (mph)": round(float(sub["speed_mph"].mean()), 2),
            "Rush-Hour Avg Speed (mph)": round(float(rush_sub["speed_mph"].mean()), 2) if len(rush_sub) > 0 else None,
            "Severe Congestion (%)": round(float(sub["severe_pct"].mean()), 2),
            "Avg Temp (C)": round(float(sub["temp_c"].mean()), 1),
            "Avg Wind (km/h)": round(float(sub["wind_kmh"].mean()), 1),
        })

    stats_df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "weather_impact.csv"
    stats_df.to_csv(out_csv, index=False)

    # Generate 300-DPI academic plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=300)

    dry = combined[combined["precip_mm"] == 0.0].groupby("hour")["speed_mph"].mean()
    rainy = combined[combined["precip_mm"] > 0.0].groupby("hour")["speed_mph"].mean()
    axes[0].plot(dry.index, dry.values, color="#2563eb", lw=2.2, label="Yağışsız / Kuru Günler")
    axes[0].plot(rainy.index, rainy.values, color="#ef4444", lw=2.2, linestyle="--", label="Yağmurlu Saatler (>0 mm)")
    axes[0].axvspan(7, 9.5, color="#f59e0b", alpha=0.09, label="Sabah Zirve")
    axes[0].axvspan(16, 19, color="#dc2626", alpha=0.09, label="Akşam Zirve")
    axes[0].set_title("Yağışın Günlük Trafik Hızı Profiline Etkisi (Gerçek LA 2012)", fontweight="bold")
    axes[0].set_xlabel("Günün Saati")
    axes[0].set_ylabel("Ağ Ortalama Hızı (mph)")
    axes[0].set_xlim(0, 24)
    axes[0].legend(loc="lower left", fontsize=8.5)

    bar_colors = ["#10b981", "#f59e0b", "#ef4444"]
    x = np.arange(len(order))
    speeds = [r["Avg Speed (mph)"] for r in rows]
    sevs = [r["Severe Congestion (%)"] for r in rows]

    bars = axes[1].bar(x - 0.18, speeds, width=0.35, color=bar_colors, alpha=0.9, label="Ortalama Hız (mph)")
    axes[1].set_ylabel("Ortalama Hız (mph)", color="#1e293b")
    axes[1].set_ylim(0, 72)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Yağışsız\n(0 mm)", "Hafif Yağmur\n(0–1 mm/sa)", "Şiddetli Yağmur\n(>1 mm/sa)"], fontsize=9)
    for b, v in zip(bars, speeds):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    ax2 = axes[1].twinx()
    bars2 = ax2.bar(x + 0.18, sevs, width=0.35, color="#7c3aed", alpha=0.75, label="Kritik Tıkanıklık (%)")
    ax2.set_ylabel("Kritik Tıkanıklık Oranı (<25 mph, %)", color="#7c3aed")
    ax2.set_ylim(0, max(sevs) * 1.45 + 2)
    for b, v in zip(bars2, sevs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"%{v:.1f}", ha="center", fontsize=8.5, color="#5b21b6", fontweight="bold")

    axes[1].set_title("Yağış Şiddetine Göre Hız Düşüşü ve Kritik Tıkanıklık Artışı", fontweight="bold")

    plt.tight_layout()
    out_plot = PLOTS_DIR / "08_weather_traffic_impact.png"
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Update experiment_results.json
    exp_path = RESULTS_DIR / "experiment_results.json"
    if exp_path.exists():
        exp_json = json.loads(exp_path.read_text(encoding="utf-8"))
        exp_json["weather_impact"] = rows
        exp_path.write_text(json.dumps(exp_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {out_plot}")
    print(f"Saved: {out_csv}")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
