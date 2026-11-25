(() => {
    const shell = document.querySelector('.app-shell');
    const API_BASE = shell?.dataset.apiBase || 'http://localhost:5100';

    let map = null;
    let markerLayer = null;
    let neighborLayer = null;
    let sensorChart = null;
    let academicMaeChart = null;
    let trainingLossChart = null;

    let currentSnapshot = null;
    let experimentsCache = null;
    let selectedSensorId = null;
    let selectedHorizon = '15';
    let selectedModel = 'gru';
    let testStartIndex = 27417;

    const statusEl = document.getElementById('systemStatus');
    const modelSelect = document.getElementById('modelSelect');
    const timeSlider = document.getElementById('timeSlider');
    const timestampLabel = document.getElementById('timestampLabel');

    // --- Tab Switching ---
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
            if (btn.dataset.tab === 'mapTab' && map) {
                setTimeout(() => map.invalidateSize(), 100);
            }
        });
    });

    // --- Map Initialization ---
    let mapFitted = false;
    function initMap() {
        map = L.map('trafficMap', { zoomControl: true }).setView([34.14, -118.28], 11);
        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 19
        }).addTo(map);
        neighborLayer = L.layerGroup().addTo(map);
        markerLayer = L.layerGroup().addTo(map);
    }

    function speedColor(speed) {
        if (speed < 25) return '#ef4444';
        if (speed < 45) return '#f59e0b';
        return '#10b981';
    }

    function getSensorSpeedForHorizon(sensor, horizon) {
        if (horizon === 'current') return sensor.currentSpeedMph;
        if (horizon === '30') return sensor.pred30MinMph;
        if (horizon === '60') return sensor.pred60MinMph;
        return sensor.pred15MinMph;
    }

    function renderMapMarkers() {
        if (!currentSnapshot || !markerLayer) return;
        markerLayer.clearLayers();

        currentSnapshot.sensors.forEach(s => {
            const spd = getSensorSpeedForHorizon(s, selectedHorizon);
            const isSelected = s.sensorId === selectedSensorId;
            const color = speedColor(spd);

            const marker = L.circleMarker([s.latitude, s.longitude], {
                radius: isSelected ? 9 : 6,
                fillColor: color,
                color: isSelected ? '#ffffff' : '#0f172a',
                weight: isSelected ? 2.5 : 1,
                fillOpacity: 0.9
            });

            marker.bindTooltip(
                `<b>Sensör #${s.sensorId}</b><br/>Şimdi: ${s.currentSpeedMph} mph<br/>+15dk: ${s.pred15MinMph} | +30dk: ${s.pred30MinMph} | +60dk: ${s.pred60MinMph}`,
                { direction: 'top' }
            );

            marker.on('click', () => {
                selectedSensorId = s.sensorId;
                renderMapMarkers();
                loadSensorDetail(s.sensorId);
            });

            marker.addTo(markerLayer);
        });

        if (!mapFitted && currentSnapshot.sensors.length > 0) {
            const bounds = L.latLngBounds(currentSnapshot.sensors.map(s => [s.latitude, s.longitude]));
            map.fitBounds(bounds, { padding: [28, 28] });
            mapFitted = true;
        }
    }

    function renderKpisAndAlerts(snap) {
        const sum = snap.networkSummary;
        document.getElementById('kpiCurrent').textContent = `${sum.avgCurrentSpeedMph} mph`;
        document.getElementById('kpiPreds').textContent = `${sum.avgPred15MinMph} / ${sum.avgPred30MinMph} / ${sum.avgPred60MinMph} mph`;
        document.getElementById('kpiSevere').textContent = `${sum.severeCount15Min} sensör`;
        document.getElementById('kpiModerate').textContent = `${sum.moderateCount15Min} sensör`;
        document.getElementById('kpiFreeFlow').textContent = `${sum.freeFlowCount15Min}`;
        timestampLabel.textContent = snap.timestamp;

        if (snap.weather) {
            const w = snap.weather;
            const icon = w.isRainy ? '🌧️' : (w.weatherCode === 0 ? '☀️' : '⛅');
            document.getElementById('kpiWeather').textContent = `${icon} ${w.temperatureC}°C · ${w.conditionTr}`;
            document.getElementById('kpiWeatherSub').textContent = `Yağış: ${w.precipitationMm} mm/sa · Rüzgar: ${w.windSpeedKmh} km/sa`;
        }

        const alertsEl = document.getElementById('alertsList');
        if (!snap.alerts || snap.alerts.length === 0) {
            alertsEl.innerHTML = '<p class="empty-msg">Bu zaman diliminde kritik tıkanıklık uyarısı yok.</p>';
            return;
        }

        alertsEl.innerHTML = snap.alerts.map(a => `
            <div class="alert-item ${a.severity === 'Severe' ? 'severe' : ''}" data-sid="${a.sensorId}" data-lat="${a.latitude}" data-lng="${a.longitude}">
                <div>
                    <div class="alert-title">Sensör #${a.sensorId} · ${a.severity === 'Severe' ? 'Kritik Tıkanıklık' : 'Orta Yoğunluk'}</div>
                    <div class="alert-meta">Şimdi: ${a.currentSpeedMph} mph → +15dk: ${a.pred15MinMph} · +30dk: ${a.pred30MinMph} · +60dk: ${a.pred60MinMph}</div>
                </div>
                <div class="alert-speed">${a.minPredictedSpeedMph} mph</div>
            </div>
        `).join('');

        alertsEl.querySelectorAll('.alert-item').forEach(item => {
            item.addEventListener('click', () => {
                const sid = item.dataset.sid;
                const lat = parseFloat(item.dataset.lat);
                const lng = parseFloat(item.dataset.lng);
                selectedSensorId = sid;
                map.setView([lat, lng], 13, { animate: true });
                renderMapMarkers();
                loadSensorDetail(sid);
            });
        });
    }

    async function loadSnapshot(timeIndex = null) {
        try {
            let url = `${API_BASE}/api/traffic/network-snapshot?model=${encodeURIComponent(selectedModel)}`;
            if (timeIndex !== null) url += `&timeIndex=${timeIndex}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            currentSnapshot = await resp.json();

            testStartIndex = currentSnapshot.testStartIndex;
            timeSlider.min = currentSnapshot.testStartIndex;
            timeSlider.max = currentSnapshot.testEndIndex;
            timeSlider.value = currentSnapshot.timeIndex;

            renderKpisAndAlerts(currentSnapshot);
            if (!selectedSensorId && currentSnapshot.alerts.length > 0) {
                selectedSensorId = currentSnapshot.alerts[0].sensorId;
            } else if (!selectedSensorId && currentSnapshot.sensors.length > 0) {
                selectedSensorId = currentSnapshot.sensors[0].sensorId;
            }
            renderMapMarkers();
            if (selectedSensorId) {
                loadSensorDetail(selectedSensorId);
            }
        } catch (err) {
            console.error('Snapshot load error:', err);
        }
    }

    async function loadSensorDetail(sensorId) {
        try {
            const url = `${API_BASE}/api/traffic/predict/${encodeURIComponent(sensorId)}?model=${encodeURIComponent(selectedModel)}&timeIndex=${timeSlider.value}&historySteps=24`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const detail = await resp.json();

            document.getElementById('sensorTitle').textContent = `Sensör #${detail.sensorId} — Geçmiş & Çoklu Ufuk Tahmini`;
            document.getElementById('sensorNeighborBadge').textContent = `Komşu Ort: ${detail.neighborAvgSpeedMph} mph (${(detail.neighbors || []).length} komşu)`;

            // Draw road-network spatial neighbor graph edges on the map (Yu et al., 2018)
            if (neighborLayer) {
                neighborLayer.clearLayers();
                (detail.neighbors || []).forEach(n => {
                    L.polyline([[detail.latitude, detail.longitude], [n.latitude, n.longitude]], {
                        color: '#38bdf8',
                        weight: 2.2,
                        opacity: 0.8,
                        dashArray: '5, 5'
                    }).bindTooltip(`Komşu Sensör #${n.sensorId} · Ağırlık: ${n.weight} · Hız: ${n.currentSpeedMph} mph`).addTo(neighborLayer);
                });
            }

            const histLabels = detail.history.map(h => h.timestamp.slice(11, 16));
            const futureLabels = [
                `${detail.actualFuture.timestamps.t15.slice(11, 16)} (+15dk)`,
                `${detail.actualFuture.timestamps.t30.slice(11, 16)} (+30dk)`,
                `${detail.actualFuture.timestamps.t60.slice(11, 16)} (+60dk)`
            ];
            const labels = [...histLabels, ...futureLabels];

            const histLen = detail.history.length;
            const histData = [...detail.history.map(h => h.speedMph), null, null, null];
            const lastVal = detail.currentSpeedMph;
            const padNulls = Array(histLen - 1).fill(null);

            const datasets = [
                {
                    label: 'Geçmiş (Gerçek)',
                    data: histData,
                    borderColor: '#e2e8f0',
                    backgroundColor: 'rgba(226, 232, 240, 0.08)',
                    borderWidth: 2,
                    tension: 0.25,
                    pointRadius: 2
                },
                {
                    label: 'Gerçek Gelecek',
                    data: [...padNulls, lastVal, detail.actualFuture.actual15MinMph, detail.actualFuture.actual30MinMph, detail.actualFuture.actual60MinMph],
                    borderColor: '#10b981',
                    borderDash: [4, 4],
                    borderWidth: 2,
                    pointRadius: 4
                }
            ];

            const modelColors = { bilstm_attention: '#a78bfa', lstm: '#3b82f6', gru: '#f59e0b' };
            const modelNames = { bilstm_attention: 'BiLSTM-Attention', lstm: 'LSTM', gru: 'GRU' };

            Object.entries(detail.predictions || {}).forEach(([mKey, mp]) => {
                datasets.push({
                    label: `${modelNames[mKey] || mKey} Tahmini`,
                    data: [...padNulls, lastVal, mp.pred15MinMph, mp.pred30MinMph, mp.pred60MinMph],
                    borderColor: modelColors[mKey] || '#60a5fa',
                    borderWidth: mKey === selectedModel ? 3 : 1.7,
                    tension: 0.2,
                    pointRadius: mKey === selectedModel ? 5 : 3
                });
            });

            if (sensorChart) sensorChart.destroy();
            sensorChart = new Chart(document.getElementById('sensorForecastChart'), {
                type: 'line',
                data: { labels, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { labels: { color: '#cbd5e1', boxWidth: 12, font: { size: 11 } } }
                    },
                    scales: {
                        x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 }, grid: { color: 'rgba(148,163,184,0.08)' } },
                        y: { min: 0, max: 75, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.08)' }, title: { display: true, text: 'mph', color: '#94a3b8' } }
                    }
                }
            });

            const activePred = detail.predictions[selectedModel];
            if (activePred) {
                document.getElementById('sensorComparisonRow').innerHTML = `
                    <div class="mini-box"><span>+15 dk Tahmin (Gerçek)</span><strong>${activePred.pred15MinMph} mph <small>(${detail.actualFuture.actual15MinMph ?? '—'})</small></strong></div>
                    <div class="mini-box"><span>+30 dk Tahmin (Gerçek)</span><strong>${activePred.pred30MinMph} mph <small>(${detail.actualFuture.actual30MinMph ?? '—'})</small></strong></div>
                    <div class="mini-box"><span>+60 dk Tahmin (Gerçek)</span><strong>${activePred.pred60MinMph} mph <small>(${detail.actualFuture.actual60MinMph ?? '—'})</small></strong></div>
                `;
            }
        } catch (err) {
            console.error('Sensor detail error:', err);
        }
    }

    async function loadExperiments() {
        try {
            const resp = await fetch(`${API_BASE}/api/traffic/experiments`);
            if (!resp.ok) return;
            const exp = await resp.json();
            if (!exp || !exp.models) return;
            experimentsCache = exp;

            document.getElementById('bestModelBadge').textContent = `En İyi Model: ${exp.best_model}`;
            if (exp.dataset) {
                document.getElementById('datasetMetaText').textContent =
                    `${exp.dataset.name} · ${exp.dataset.num_sensors} Sensör × ${exp.dataset.num_timesteps.toLocaleString()} Zaman Adımı (${exp.dataset.date_start.slice(0,10)} – ${exp.dataset.date_end.slice(0,10)})`;
            }

            // Fill comparison table
            const tbody = document.querySelector('#modelComparisonTable tbody');
            tbody.innerHTML = Object.entries(exp.models).map(([name, info]) => {
                const h = info.metrics.horizons;
                const ov = info.metrics.overall;
                const isBest = name === exp.best_model;
                return `
                    <tr class="${isBest ? 'best-row' : ''}">
                        <td><strong>${name}</strong> ${isBest ? '★' : ''}</td>
                        <td>${info.num_params ? info.num_params.toLocaleString() : '—'}</td>
                        <td>${h['15 min'].mae.toFixed(2)} / ${h['15 min'].rmse.toFixed(2)}</td>
                        <td>${h['30 min'].mae.toFixed(2)} / ${h['30 min'].rmse.toFixed(2)}</td>
                        <td>${h['60 min'].mae.toFixed(2)} / ${h['60 min'].rmse.toFixed(2)}</td>
                        <td><strong>${ov.mae.toFixed(3)}</strong></td>
                        <td>${ov.rmse.toFixed(3)}</td>
                        <td>${ov.mape.toFixed(2)}%</td>
                        <td>${ov.r2.toFixed(4)}</td>
                    </tr>
                `;
            }).join('');

            // Fill ablation table
            const abBody = document.querySelector('#ablationTable tbody');
            abBody.innerHTML = (exp.ablation_study || []).map(r => `
                <tr>
                    <td><strong>${r.experiment}</strong></td>
                    <td>${r.num_layers}</td>
                    <td>${r.hidden_size}</td>
                    <td>${r.use_spatial ? 'Evet (Komşu Ağı)' : 'Hayır (Sadece Zaman)'}</td>
                    <td>${r.num_params.toLocaleString()}</td>
                    <td><strong>${r.metrics.overall.mae.toFixed(3)}</strong></td>
                    <td>${r.metrics.overall.rmse.toFixed(3)}</td>
                    <td>${r.metrics.overall.mape.toFixed(2)}%</td>
                    <td>${r.metrics.overall.r2.toFixed(4)}</td>
                </tr>
            `).join('');

            // Academic MAE Grouped Bar Chart
            const colors = {
                'Persistence': '#64748b',
                'Historical Avg': '#94a3b8',
                'LSTM': '#3b82f6',
                'GRU': '#10b981',
                'BiLSTM-Attention': '#8b5cf6'
            };
            const barDatasets = Object.entries(exp.models).map(([name, info]) => ({
                label: name,
                data: [
                    info.metrics.horizons['15 min'].mae,
                    info.metrics.horizons['30 min'].mae,
                    info.metrics.horizons['60 min'].mae
                ],
                backgroundColor: colors[name] || '#60a5fa',
                borderRadius: 5
            }));

            if (academicMaeChart) academicMaeChart.destroy();
            academicMaeChart = new Chart(document.getElementById('academicMaeChart'), {
                type: 'bar',
                data: { labels: ['15 Dakika', '30 Dakika', '60 Dakika'], datasets: barDatasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#cbd5e1' } } },
                    scales: {
                        x: { ticks: { color: '#cbd5e1' }, grid: { display: false } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.1)' }, title: { display: true, text: 'Masked MAE (mph)', color: '#94a3b8' } }
                    }
                }
            });

            // Validation Loss Curve Chart
            const lossDatasets = [];
            let maxEpochs = 0;
            Object.entries(exp.models).forEach(([name, info]) => {
                if (info.history && info.history.val_loss) {
                    maxEpochs = Math.max(maxEpochs, info.history.val_loss.length);
                    lossDatasets.push({
                        label: `${name} (Val MAE)`,
                        data: info.history.val_loss,
                        borderColor: colors[name] || '#60a5fa',
                        borderWidth: 2.2,
                        tension: 0.2
                    });
                }
            });

            if (trainingLossChart) trainingLossChart.destroy();
            trainingLossChart = new Chart(document.getElementById('trainingLossChart'), {
                type: 'line',
                data: {
                    labels: Array.from({ length: maxEpochs }, (_, i) => `Epoch ${i + 1}`),
                    datasets: lossDatasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#cbd5e1' } } },
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.08)' } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148,163,184,0.08)' }, title: { display: true, text: 'Validation MAE (mph)', color: '#94a3b8' } }
                    }
                }
            });

            renderDynamicConfusionMatrix();
        } catch (err) {
            console.error('Experiments load error:', err);
        }
    }

    function renderDynamicConfusionMatrix() {
        const cmSelect = document.getElementById('cmModelSelect');
        const hSelect = document.getElementById('cmHorizonSelect');
        const tbody = document.querySelector('#dynamicCmTable tbody');
        const kpiRow = document.getElementById('cmKpiRow');
        if (!experimentsCache || !experimentsCache.confusion_matrices || !tbody) return;

        const mName = cmSelect.value;
        const hName = hSelect.value;
        const entry = experimentsCache.confusion_matrices[mName]?.[hName];
        if (!entry) {
            tbody.innerHTML = '<tr><td colspan="4" class="empty-msg">Bu model için karmaşıklık matrisi henüz üretilmedi.</td></tr>';
            kpiRow.innerHTML = '';
            return;
        }

        kpiRow.innerHTML = `
            <div class="mini-box"><span>Genel Doğruluk (Accuracy)</span><strong>%${entry.accuracy_pct}</strong></div>
            <div class="mini-box"><span>Macro F1-Score</span><strong>${entry.macro_f1}</strong></div>
            <div class="mini-box"><span>Weighted F1-Score</span><strong>${entry.weighted_f1}</strong></div>
            <div class="mini-box"><span>Kritik Tıkanıklık Recall / Prec</span><strong>${entry.severe_recall} / ${entry.severe_precision}</strong></div>
        `;

        tbody.innerHTML = entry.labels.map((rowLabel, i) => {
            const cells = entry.matrix[i].map((count, j) => {
                const pct = entry.matrix_pct[i][j];
                const alpha = Math.max(0.08, Math.min(0.75, pct / 100));
                const bg = i === j
                    ? `rgba(16, 185, 129, ${alpha})`
                    : `rgba(239, 68, 68, ${alpha * 0.7})`;
                return `<td style="background:${bg}">${count.toLocaleString()}<span class="cm-cell-pct">(${pct}%)</span></td>`;
            }).join('');
            return `<tr><td class="cm-row-label">${rowLabel}</td>${cells}</tr>`;
        }).join('');
    }

    document.getElementById('cmModelSelect')?.addEventListener('change', renderDynamicConfusionMatrix);
    document.getElementById('cmHorizonSelect')?.addEventListener('change', renderDynamicConfusionMatrix);

    function renderPlotsGallery() {
        const plots = [
            { file: '01_eda_daily_weekly_pattern.png', title: '1. METR-LA Günlük Zirve Saatler ve Hız Dağılımı (EDA)' },
            { file: '02_training_validation_curves.png', title: '2. LSTM / GRU / BiLSTM-Attention Eğitim & Doğrulama Eğrileri' },
            { file: '03_model_comparison_horizons.png', title: '3. 15 / 30 / 60 Dakika Ufuklarına Göre MAE ve RMSE Karşılaştırması' },
            { file: '04_ablation_hyperparameters.png', title: '4. Hiperparametre & Uzamsal Komşuluk Özelliği Ablasyonu' },
            { file: '05_actual_vs_predicted_timeseries.png', title: '5. 48 Saatlik Gerçek vs. Tahmin Zaman Serisi (Test Seti)' },
            { file: '06_error_distribution_scatter.png', title: '6. En İyi Model Gerçek-Tahmin Korelasyonu (R²) ve Hata Dağılımı' },
            { file: '07_congestion_confusion_matrix.png', title: '7. Karar Destek Sistemi Tıkanıklık Sınıflandırma Karmaşıklık Matrisi (Confusion Matrix)' },
            { file: '08_weather_traffic_impact.png', title: '8. Gerçek LA Meteoroloji Verisine Göre Yağışın Trafik Hızına Etkisi (Open-Meteo 2012)' }
        ];
        const gallery = document.getElementById('plotsGallery');
        gallery.innerHTML = plots.map(p => `
            <div class="plot-card">
                <a href="${API_BASE}/api/traffic/plots/${p.file}" target="_blank" rel="noopener">
                    <img src="${API_BASE}/api/traffic/plots/${p.file}" alt="${p.title}" loading="lazy" />
                </a>
                <div class="plot-caption">${p.title}</div>
            </div>
        `).join('');
    }

    async function checkHealth() {
        try {
            const resp = await fetch(`${API_BASE}/api/traffic/health`);
            if (!resp.ok) throw new Error();
            const h = await resp.json();
            const count = h.aiService?.loadedModels?.length || 0;
            statusEl.textContent = `Çevrimiçi · ${count} AI Modeli Hazır`;
            statusEl.className = 'status-pill status-ok';
            return true;
        } catch {
            statusEl.textContent = 'AI / API Bağlantısı Bekleniyor';
            statusEl.className = 'status-pill status-err';
            return false;
        }
    }

    // --- Event Listeners ---
    modelSelect.addEventListener('change', () => {
        selectedModel = modelSelect.value;
        loadSnapshot(parseInt(timeSlider.value, 10));
    });

    document.querySelectorAll('#horizonButtons .seg-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#horizonButtons .seg-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedHorizon = btn.dataset.horizon;
            renderMapMarkers();
        });
    });

    let sliderTimer = null;
    timeSlider.addEventListener('input', () => {
        clearTimeout(sliderTimer);
        sliderTimer = setTimeout(() => loadSnapshot(parseInt(timeSlider.value, 10)), 180);
    });

    document.querySelectorAll('.chip-btn[data-offset]').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = testStartIndex + parseInt(btn.dataset.offset, 10);
            timeSlider.value = target;
            loadSnapshot(target);
        });
    });

    // --- Custom Real-Time POST Inference ---
    document.getElementById('runCustomPredictBtn')?.addEventListener('click', async () => {
        const raw = document.getElementById('customSpeedsInput').value;
        const speeds = raw.split(',').map(v => parseFloat(v.trim())).filter(v => !Number.isNaN(v));
        const outEl = document.getElementById('customPredictOutput');
        if (speeds.length !== 12) {
            outEl.innerHTML = `<div class="mini-box" style="grid-column:1/-1;border-color:#ef4444"><span>Hata</span><strong>Tam olarak 12 adet hız değeri (son 60 dk) girmelisiniz. Şu an: ${speeds.length}</strong></div>`;
            return;
        }
        const payload = {
            speeds,
            hour: parseInt(document.getElementById('customHourInput').value, 10) || 8,
            minute: 0,
            dayOfWeek: parseInt(document.getElementById('customDaySelect').value, 10) || 2,
            model: selectedModel
        };
        try {
            const resp = await fetch(`${API_BASE}/api/traffic/predict/custom`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const labels = { gru: 'GRU (En İyi)', lstm: 'LSTM', bilstm_attention: 'BiLSTM-Attention' };
            outEl.innerHTML = Object.entries(data.allModels).map(([k, v]) => `
                <div class="mini-box">
                    <span>${labels[k] || k}</span>
                    <strong>+15dk: ${v.pred15MinMph} · +30dk: ${v.pred30MinMph} · +60dk: ${v.pred60MinMph} mph</strong>
                    <span style="margin-top:4px">Durum: ${v.status15Min} → ${v.status30Min} → ${v.status60Min}</span>
                </div>
            `).join('');
        } catch (err) {
            outEl.innerHTML = `<div class="mini-box" style="grid-column:1/-1;border-color:#ef4444"><span>İstek Hatası</span><strong>${err.message}</strong></div>`;
        }
    });

    // --- Bootstrap ---
    initMap();
    renderPlotsGallery();
    checkHealth().then(ok => {
        if (ok) {
            loadSnapshot();
            loadExperiments();
        }
    });
})();
