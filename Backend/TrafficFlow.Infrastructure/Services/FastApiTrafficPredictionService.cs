using System.Net.Http.Json;
using System.Text.Json;
using TrafficFlow.Core.DTOs;
using TrafficFlow.Core.Interfaces;

namespace TrafficFlow.Infrastructure.Services;

public class FastApiTrafficPredictionService : ITrafficPredictionService
{
    private readonly HttpClient _http;
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public FastApiTrafficPredictionService(HttpClient http)
    {
        _http = http;
    }

    public async Task<AiHealthDto?> GetHealthAsync(CancellationToken ct = default)
    {
        return await _http.GetFromJsonAsync<AiHealthDto>("/health", JsonOpts, ct);
    }

    public async Task<IReadOnlyList<SensorLocationDto>> GetSensorsAsync(CancellationToken ct = default)
    {
        var result = await _http.GetFromJsonAsync<List<SensorLocationDto>>("/api/ai/sensors", JsonOpts, ct);
        return result ?? [];
    }

    public async Task<NetworkSnapshotDto?> GetNetworkSnapshotAsync(string model, int? timeIndex, CancellationToken ct = default)
    {
        var url = $"/api/ai/network-snapshot?model={Uri.EscapeDataString(model)}";
        if (timeIndex.HasValue)
            url += $"&time_index={timeIndex.Value}";
        return await _http.GetFromJsonAsync<NetworkSnapshotDto>(url, JsonOpts, ct);
    }

    public async Task<SensorForecastDetailDto?> GetSensorForecastAsync(
        string sensorId,
        string model,
        int? timeIndex,
        int historySteps,
        CancellationToken ct = default)
    {
        var url = $"/api/ai/predict/{Uri.EscapeDataString(sensorId)}?model={Uri.EscapeDataString(model)}&history_steps={historySteps}";
        if (timeIndex.HasValue)
            url += $"&time_index={timeIndex.Value}";
        return await _http.GetFromJsonAsync<SensorForecastDetailDto>(url, JsonOpts, ct);
    }

    public async Task<CustomPredictResponseDto?> PredictCustomAsync(CustomPredictRequestDto request, CancellationToken ct = default)
    {
        using var resp = await _http.PostAsJsonAsync("/api/ai/predict/custom", request, ct);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<CustomPredictResponseDto>(JsonOpts, ct);
    }

    public async Task<JsonElement?> GetExperimentResultsAsync(CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync("/api/ai/experiments", ct);
        resp.EnsureSuccessStatusCode();
        await using var stream = await resp.Content.ReadAsStreamAsync(ct);
        using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
        return doc.RootElement.Clone();
    }
}
