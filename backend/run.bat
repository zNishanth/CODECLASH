@echo off
title SPACEIND - Green Building Digital Twin
echo ========================================================
echo   SPACEIND // GREEN BUILDING DIGITAL TWIN PLATFORM
echo   Industrial Brutalist Telemetry, 3D BIM ^& Facility AI
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [INFO] Starting Green Building Digital Twin on http://127.0.0.1:8000 ...
python server.py
pause
