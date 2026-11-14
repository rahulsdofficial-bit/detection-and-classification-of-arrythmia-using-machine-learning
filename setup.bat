@echo off
echo ====================================================================
echo   ARRHYTHMIA DETECTION & CLASSIFICATION SYSTEM - AUTOMATIC SETUP
echo ====================================================================

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10+ and re-run setup.bat.
    pause
    exit /b 1
)

echo.
echo [1/4] Installing Required Python Dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/4] Verifying / Downloading MIT-BIH Arrhythmia Dataset...
python download_dataset.py

echo.
echo [3/4] Checking and Training Machine Learning Models (WKNN)...
python train_model.py

echo.
echo [4/4] Setting Up Application Database & Environment...
if not exist .env (
    copy .env.example .env >nul
    echo Created .env configuration from .env.example.
)

echo.
echo ====================================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo   You can now launch the web application by running: run.bat
echo ====================================================================
pause
