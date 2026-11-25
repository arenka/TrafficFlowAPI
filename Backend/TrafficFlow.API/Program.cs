using TrafficFlow.Core.Interfaces;
using TrafficFlow.Infrastructure.Options;
using TrafficFlow.Infrastructure.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new()
    {
        Title = "METR-LA Dinamik Trafik Akışı Tahmini Web API",
        Version = "v1",
        Description = "ASP.NET Core 9 Clean Architecture REST API — PyTorch LSTM / GRU / BiLSTM-Attention Mikroservis Entegrasyonu"
    });
});

builder.Services.Configure<AiServiceOptions>(
    builder.Configuration.GetSection(AiServiceOptions.SectionName));

builder.Services.AddHttpClient<ITrafficPredictionService, FastApiTrafficPredictionService>((sp, client) =>
{
    var opts = builder.Configuration.GetSection(AiServiceOptions.SectionName).Get<AiServiceOptions>() ?? new AiServiceOptions();
    client.BaseAddress = new Uri(opts.BaseUrl);
    client.Timeout = TimeSpan.FromSeconds(opts.TimeoutSeconds);
});

builder.Services.AddCors(options =>
{
    options.AddPolicy("DashboardCors", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "TrafficFlow API v1");
        c.RoutePrefix = "swagger";
    });
}

app.UseCors("DashboardCors");
app.MapControllers();

app.Run();
