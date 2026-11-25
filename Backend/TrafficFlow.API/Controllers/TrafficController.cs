using Microsoft.AspNetCore.Mvc;
using TrafficFlow.Core.DTOs;
using TrafficFlow.Core.Interfaces;

namespace TrafficFlow.API.Controllers;

[ApiController]
[Route("api/traffic")]
public class TrafficController : ControllerBase
{
    private readonly ITrafficPredictionService _predictionService;
    private readonly IConfiguration _config;
    private readonly ILogger<TrafficController> _logger;

    private static readonly HashSet<string> AllowedModels = new(StringComparer.OrdinalIgnoreCase)
    {
        "lstm", "gru", "bilstm_attention"
    };

    public TrafficController(
        ITrafficPredictionService predictionService,
        IConfiguration config,
        ILogger<TrafficController> logger)
    {
        _predictionService = predictionService;
        _config = config;
        _logger = logger;
    }

    [HttpGet("health")]
    public async Task<IActionResult> GetHealth(CancellationToken ct)
    {
        try
        {
            var aiHealth = await _predictionService.GetHealthAsync(ct);
            return Ok(new
            {
                apiStatus = "healthy",
                aiService = aiHealth
            });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "AI inference service unreachable.");
            return StatusCode(StatusCodes.Status503ServiceUnavailable, new
            {
                apiStatus = "healthy",
                aiStatus = "unreachable",
                detail = ex.Message
            });
        }
    }

    [HttpGet("sensors")]
    public async Task<IActionResult> GetSensors(CancellationToken ct)
    {
        var sensors = await _predictionService.GetSensorsAsync(ct);
        return Ok(sensors);
    }

    [HttpGet("network-snapshot")]
    public async Task<IActionResult> GetNetworkSnapshot(
        [FromQuery] string model = "bilstm_attention",
        [FromQuery] int? timeIndex = null,
        CancellationToken ct = default)
    {
        if (!AllowedModels.Contains(model))
            return BadRequest(new { error = $"Invalid model '{model}'. Allowed: lstm, gru, bilstm_attention." });

        var snapshot = await _predictionService.GetNetworkSnapshotAsync(model.ToLowerInvariant(), timeIndex, ct);
        return snapshot is null ? NotFound() : Ok(snapshot);
    }

    [HttpGet("predict/{sensorId}")]
    public async Task<IActionResult> GetSensorForecast(
        string sensorId,
        [FromQuery] string model = "bilstm_attention",
        [FromQuery] int? timeIndex = null,
        [FromQuery] int historySteps = 36,
        CancellationToken ct = default)
    {
        if (!AllowedModels.Contains(model))
            return BadRequest(new { error = $"Invalid model '{model}'. Allowed: lstm, gru, bilstm_attention." });

        historySteps = Math.Clamp(historySteps, 12, 144);
        var detail = await _predictionService.GetSensorForecastAsync(sensorId, model.ToLowerInvariant(), timeIndex, historySteps, ct);
        return detail is null ? NotFound() : Ok(detail);
    }

    /// <summary>
    /// Real-time inference endpoint accepting 12 external 5-minute sensor speed readings (last 60 min).
    /// </summary>
    [HttpPost("predict/custom")]
    public async Task<IActionResult> PredictCustom([FromBody] CustomPredictRequestDto request, CancellationToken ct)
    {
        if (request.Speeds is null || request.Speeds.Count != 12)
            return BadRequest(new { error = "Exactly 12 historical 5-minute speed values (last 60 minutes) are required." });

        if (!AllowedModels.Contains(request.Model))
            return BadRequest(new { error = $"Invalid model '{request.Model}'. Allowed: lstm, gru, bilstm_attention." });

        var normalized = request with { Model = request.Model.ToLowerInvariant() };
        var result = await _predictionService.PredictCustomAsync(normalized, ct);
        return result is null ? StatusCode(StatusCodes.Status502BadGateway) : Ok(result);
    }

    [HttpGet("experiments")]
    public async Task<IActionResult> GetExperiments(CancellationToken ct)
    {
        var experiments = await _predictionService.GetExperimentResultsAsync(ct);
        return experiments is null ? NotFound() : Ok(experiments);
    }

    [HttpGet("plots/{fileName}")]
    public IActionResult GetPlotImage(string fileName)
    {
        var safeName = Path.GetFileName(fileName);
        if (string.IsNullOrWhiteSpace(safeName) || !safeName.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            return BadRequest(new { error = "Only PNG plot files are supported." });

        var plotsDir = _config["AiService:PlotsDirectory"]
            ?? Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "AI_Models", "plots"));

        var fullPath = Path.Combine(plotsDir, safeName);
        if (!System.IO.File.Exists(fullPath))
            return NotFound(new { error = $"Plot '{safeName}' not found." });

        return PhysicalFile(fullPath, "image/png");
    }
}
