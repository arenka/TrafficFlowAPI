# Zaman Serisi Analizi ve Derin Öğrenme Yöntemleriyle Dinamik Trafik Akışı Tahmini

Yüksek lisans bitirme projesi kapsamında geliştirilen, **METR-LA** (Los Angeles 207 otoyol sensörü) veri seti üzerinde çalışan uçtan uca **Spatio-Temporal Derin Öğrenme + Mikroservis Karar Destek Sistemi**.

## Mimari Genel Bakış

```text
[METR-LA Gerçek Veri Seti (34.272 × 207)]
        │
        ▼
[1. AI & Modelleme Katmanı — Python / PyTorch / FastAPI (:8000)]
   • Baseline: Persistence, Historical Average (HA)
   • Deep Learning: LSTM, GRU, BiLSTM + Temporal Attention
   • Çoklu Tahmin Ufku: +15 dk, +30 dk, +60 dk
   • Uzamsal Komşuluk (Gaussian Adjacency Matrix) + Döngüsel Zaman Özellikleri
   • Hiperparametre & Uzamsal Özellik Ablasyonu + 300 DPI Tez Grafikleri
        │  (HTTP / JSON Mikroservis)
        ▼
[2. Servis Katmanı — ASP.NET Core 9 Web API (:5100)]
   • Clean Architecture: TrafficFlow.Core / TrafficFlow.Infrastructure / TrafficFlow.API
   • REST Endpoints: /api/traffic/health, /sensors, /network-snapshot, /predict/{id}, /experiments, /plots/{name}
        │  (REST / JSON)
        ▼
[3. Sunum Katmanı — ASP.NET Core 9 MVC + JavaScript (:5200)]
   • Leaflet.js İnteraktif Los Angeles 207 Sensör Yoğunluk Haritası
   • Chart.js Çoklu Model & Çoklu Ufuk Zaman Serisi Grafikleri
   • Karar Destek Kritik Tıkanıklık Uyarı Paneli
   • Akademik Model Karşılaştırma, Ablasyon Tablosu ve 300 DPI Grafik Galerisi
```

## Klasör Yapısı

- `AI_Models/data/` — Gerçek `metr-la.h5`, `graph_sensor_locations.csv`, `distances_la_2012.csv`
- `AI_Models/src/` — `config.py`, `data_loader.py`, `metrics.py`, `models.py`, `plots_generator.py`, `train_experiments.py`, `inference_api.py`
- `AI_Models/models/` — Eğitilen PyTorch ağırlıkları (`lstm_best.pth`, `gru_best.pth`, `bilstm_attention_best.pth`)
- `AI_Models/results/` — `experiment_results.json`, `model_comparison.csv`, `ablation_study.csv`
- `AI_Models/plots/` — 6 adet 300 DPI tez grafiği
- `Backend/TrafficFlow.Core/` — Domain DTO'ları ve servis arayüzleri
- `Backend/TrafficFlow.Infrastructure/` — FastAPI HTTP istemci entegrasyonu
- `Backend/TrafficFlow.API/` — ASP.NET Core 9 Web API
- `Backend/TrafficFlow.Web/` — ASP.NET Core 9 MVC Karar Destek Dashboard'u

## Çalıştırma

### Tek Komutla Tüm Sistemi Başlatma (PowerShell)
```powershell
.\start_all.ps1
```
Ardından tarayıcıda **http://localhost:5200** adresini açın.

### Modelleri Yeniden Eğitme ve Deneyleri Tekrarlama
```powershell
.\AI_Models\venv\Scripts\python.exe .\AI_Models\src\train_experiments.py
```
