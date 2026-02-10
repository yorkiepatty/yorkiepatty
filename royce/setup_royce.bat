@echo off
:: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
::  Royce AI Assistant — Windows 11 Setup
:: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo.
echo ====================================================
echo   ROYCE AI ASSISTANT — Setup
echo   Confident. Smart. Caring.
echo ====================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/5] Installing core dependencies...
pip install anthropic requests beautifulsoup4 python-dotenv pygame SpeechRecognition
pip install lxml numpy

echo.
echo [3/5] Installing voice/audio dependencies...
pip install sounddevice gTTS boto3

echo.
echo [4/5] Installing optional dependencies...
pip install yt-dlp 2>nul
pip install Pillow 2>nul

echo.
echo [5/5] Setting up directories...
if not exist "data" mkdir data
if not exist "data\memory" mkdir data\memory
if not exist "data\logs" mkdir data\logs
if not exist "data\images" mkdir data\images
if not exist "data\music_cache" mkdir data\music_cache

:: Check for .env
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo.
        echo [IMPORTANT] Created .env from .env.example
        echo Edit .env and add your API keys before running Royce.
    )
)

echo.
echo ====================================================
echo   Setup Complete!
echo.
echo   To start Royce:
echo     python -m royce.launcher              (text mode)
echo     python -m royce.launcher --mode voice  (voice mode)
echo     python -m royce.launcher --mode hybrid (voice + text)
echo.
echo   Don't forget to edit .env with your API keys!
echo ====================================================
echo.
pause
