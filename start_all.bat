@echo off
title METR-LA Traffic Flow Decision Support System Launcher
cd /d "%~dp0"

echo [1/3] Starting Python FastAPI AI Microservice (http://127.0.0.1:8000)...
start "1 - AI Microservice (FastAPI :8000)" cmd /k "cd /d "%~dp0AI_Models\src" && "%~dp0AI_Models\venv\Scripts\uvicorn.exe" inference_api:app --host 127.0.0.1 --port 8000"

timeout /t 4 /nobreak >nul

echo [2/3] Starting ASP.NET Core 9 Web API (http://localhost:5100)...
start "2 - ASP.NET Core Web API (:5100)" cmd /k "cd /d "%~dp0Backend\TrafficFlow.API" && dotnet run --launch-profile http"

timeout /t 3 /nobreak >nul

echo [3/3] Starting ASP.NET Core 9 MVC Dashboard (http://localhost:5200)...
start "3 - ASP.NET Core MVC Dashboard (:5200)" cmd /k "cd /d "%~dp0Backend\TrafficFlow.Web" && dotnet run --launch-profile http"

echo.
echo Tum servisler baslatildi! Tarayicida http://localhost:5200 adresini acin.
pause
