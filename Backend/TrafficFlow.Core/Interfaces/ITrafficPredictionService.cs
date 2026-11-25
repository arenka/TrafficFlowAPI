using System.Text.Json;
using TrafficFlow.Core.DTOs;

namespace TrafficFlow.Core.Interfaces;

public interface ITrafficPredictionService
{
    Task<AiHealthDto?> GetHealthAsync(CancellationToken ct = default);
    Task<IReadOnlyList<SensorLocationDto>> GetSensorsAsync(CancellationToken ct = default);
    Task<NetworkSnapshotDto?> GetNetworkSnapshotAsync(string model, int? timeIndex, CancellationToken ct = default);
    Task<SensorForecastDetailDto?> GetSensorForecastAsync(string sensorId, string model, int? timeIndex, int historySteps, CancellationToken ct = default);
    Task<CustomPredictResponseDto?> PredictCustomAsync(CustomPredictRequestDto request, CancellationToken ct = default);
    Task<JsonElement?> GetExperimentResultsAsync(CancellationToken ct = default);
}
