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

:: ─── Step 1: Create venv ──────────────────────────────────────
echo [1/6] Setting up virtual environment...
if not exist "..\royce_env" (
    python -m venv ..\royce_env
    echo       Created royce_env
) else (
    echo       royce_env already exists, reusing it
)

:: Activate the venv
call ..\royce_env\Scripts\activate.bat
echo       Activated royce_env
echo.

:: ─── Step 2: Upgrade pip ─────────────────────────────────────
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
:: ─── Step 3: Core dependencies ────────────────────────────────
echo [3/6] Installing core dependencies...
pip install anthropic requests beautifulsoup4 python-dotenv pygame SpeechRecognition
pip install lxml numpy

echo.
:: ─── Step 4: Voice/audio dependencies ─────────────────────────
echo [4/6] Installing voice/audio dependencies...
pip install sounddevice gTTS boto3

echo.
:: ─── Step 5: Optional dependencies ────────────────────────────
echo [5/6] Installing optional dependencies...
pip install yt-dlp 2>nul
pip install Pillow 2>nul

echo.
:: ─── Step 6: Directories and .env ─────────────────────────────
echo [6/6] Setting up directories...
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

:: ─── Create a quick-launch script ─────────────────────────────
echo @echo off > ..\run_royce.bat
echo call "%%~dp0royce_env\Scripts\activate.bat" >> ..\run_royce.bat
echo cd "%%~dp0" >> ..\run_royce.bat
echo python -m royce.launcher %%* >> ..\run_royce.bat

echo.
echo ====================================================
echo   Setup Complete!
echo.
echo   To start Royce, just double-click:
echo     run_royce.bat
echo.
echo   Or from a terminal:
echo     run_royce.bat                  (text mode)
echo     run_royce.bat --mode voice     (voice mode)
echo     run_royce.bat --mode hybrid    (voice + text)
echo.
echo   Don't forget to edit royce\.env with your API keys!
echo ====================================================
echo.
pause
