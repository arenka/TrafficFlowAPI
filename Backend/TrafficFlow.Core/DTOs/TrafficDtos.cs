namespace TrafficFlow.Core.DTOs;

public record SensorLocationDto(
    string SensorId,
    int Index,
    double Latitude,
    double Longitude,
    double MeanSpeedMph);

public record NetworkSummaryDto(
    double AvgCurrentSpeedMph,
    double AvgPred15MinMph,
    double AvgPred30MinMph,
    double AvgPred60MinMph,
    int SevereCount15Min,
    int ModerateCount15Min,
    int FreeFlowCount15Min);

public record CongestionAlertDto(
    string SensorId,
    double Latitude,
    double Longitude,
    double CurrentSpeedMph,
    double MinPredictedSpeedMph,
    double Pred15MinMph,
    double Pred30MinMph,
    double Pred60MinMph,
    string Severity,
    double SpeedDropMph);

public record SensorSnapshotDto(
    string SensorId,
    double Latitude,
    double Longitude,
    double CurrentSpeedMph,
    double Pred15MinMph,
    double Pred30MinMph,
    double Pred60MinMph,
    double? Actual15MinMph,
    double? Actual30MinMph,
    double? Actual60MinMph,
    string StatusCurrent,
    string Status15Min,
    string Status30Min,
    string Status60Min);

public record WeatherInfoDto(
    double TemperatureC,
    double PrecipitationMm,
    double WindSpeedKmh,
    int WeatherCode,
    string ConditionTr,
    bool IsRainy);

public record NetworkSnapshotDto(
    string Model,
    int TimeIndex,
    string Timestamp,
    int TestStartIndex,
    int TestEndIndex,
    WeatherInfoDto? Weather,
    NetworkSummaryDto NetworkSummary,
    List<CongestionAlertDto> Alerts,
    List<SensorSnapshotDto> Sensors);

public record HistoryPointDto(string Timestamp, double SpeedMph);

public record ModelHorizonPredictionDto(
    double Pred15MinMph,
    double Pred30MinMph,
    double Pred60MinMph);

public record ActualFutureTimestampsDto(string T15, string T30, string T60);

public record ActualFutureDto(
    double? Actual15MinMph,
    double? Actual30MinMph,
    double? Actual60MinMph,
    ActualFutureTimestampsDto Timestamps);

public record SpatialNeighborDto(
    string SensorId,
    double Latitude,
    double Longitude,
    double Weight,
    double CurrentSpeedMph);

public record SensorForecastDetailDto(
    string SensorId,
    double Latitude,
    double Longitude,
    string SelectedModel,
    int TimeIndex,
    string Timestamp,
    double CurrentSpeedMph,
    double NeighborAvgSpeedMph,
    List<SpatialNeighborDto> Neighbors,
    List<HistoryPointDto> History,
    Dictionary<string, ModelHorizonPredictionDto> Predictions,
    ActualFutureDto ActualFuture);

public record CustomPredictRequestDto(
    List<double> Speeds,
    List<double>? NeighborSpeeds = null,
    int Hour = 8,
    int Minute = 30,
    int DayOfWeek = 2,
    string Model = "gru");

public record CustomModelPredictionDto(
    double Pred15MinMph,
    double Pred30MinMph,
    double Pred60MinMph,
    string Status15Min,
    string Status30Min,
    string Status60Min);

public record CustomPredictResponseDto(
    string SelectedModel,
    double InputCurrentSpeedMph,
    CustomModelPredictionDto PrimaryPrediction,
    Dictionary<string, CustomModelPredictionDto> AllModels);

public record AiHealthDto(
    string Status,
    List<string> LoadedModels,
    int NumSensors,
    int TestStartIndex,
    int TestEndIndex);
