# One-Click Startup Script: AI Microservice + ASP.NET Core Web API + MVC Dashboard
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "1/3 Starting Python FastAPI AI Microservice (http://127.0.0.1:8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\AI_Models\src'; & '$root\AI_Models\venv\Scripts\uvicorn.exe' inference_api:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 3

Write-Host "2/3 Starting ASP.NET Core 9 Web API (http://localhost:5100)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\Backend\TrafficFlow.API'; dotnet run --launch-profile http"

Start-Sleep -Seconds 2

Write-Host "3/3 Starting ASP.NET Core 9 MVC Decision Support Dashboard (http://localhost:5200)..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\Backend\TrafficFlow.Web'; dotnet run --launch-profile http"

Write-Host "`nAll 3 services launched! Open http://localhost:5200 in your browser." -ForegroundColor Yellow
