@echo off
title PlantCure v2
echo.
echo  ============================================
echo    PlantCure v2 - Plant Disease Detection
echo  ============================================
echo.

python --version >nul 2>&1
if errorlevel 1 ( echo [ERROR] Python not found. Download from python.org & pause & exit /b 1 )

if not exist ".venv" (
    echo [SETUP] Creating virtual environment ...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [SETUP] Installing dependencies ...
pip install -r requirements.txt -q

if exist "models\combined_plant_disease_model" (
    echo [INFO]  Trained model found.
) else (
    echo [INFO]  No model — running in Demo Mode.
)

echo.
echo [START] Opening http://localhost:5000
echo         Press Ctrl+C to stop the server.
echo.
python app.py
pause
