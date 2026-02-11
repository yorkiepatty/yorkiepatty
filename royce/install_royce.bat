@echo off
setlocal enabledelayedexpansion
:: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
::  Royce AI Assistant — One-Click Installer for Windows 11
::  Just double-click this file. It handles everything.
:: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

title Royce AI Assistant — Installer
color 0F

echo.
echo  ====================================================
echo.
echo   ROYCE AI ASSISTANT
echo   Confident. Smart. Caring.
echo.
echo   One-Click Installer for Windows 11
echo.
echo  ====================================================
echo.
echo  This will set up everything Royce needs to run.
echo  It won't touch anything else on your computer.
echo.
pause

:: ─── Step 1: Check Python ─────────────────────────────────────
echo.
echo  [1/8] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  =====================================================
    echo   Python is not installed!
    echo.
    echo   Royce needs Python to run. Here's what to do:
    echo.
    echo   1. Go to https://www.python.org/downloads/
    echo   2. Click the big yellow "Download Python" button
    echo   3. Run the installer
    echo   4. IMPORTANT: Check "Add Python to PATH" at the bottom
    echo   5. Click "Install Now"
    echo   6. After it's done, run this installer again
    echo  =====================================================
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo         Found %PYVER%

:: ─── Step 2: Pick install location ────────────────────────────
echo.
echo  [2/8] Setting up install location...
set "INSTALL_DIR=%USERPROFILE%\Royce"
echo         Installing to: %INSTALL_DIR%

if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
    echo         Created folder
) else (
    echo         Folder already exists, updating...
)

:: Copy all royce files to install directory
set "SCRIPT_DIR=%~dp0"
xcopy "%SCRIPT_DIR%*.py" "%INSTALL_DIR%\royce\" /E /I /Y /Q >nul 2>&1
xcopy "%SCRIPT_DIR%*.json" "%INSTALL_DIR%\royce\" /E /I /Y /Q >nul 2>&1
xcopy "%SCRIPT_DIR%*.txt" "%INSTALL_DIR%\royce\" /E /I /Y /Q >nul 2>&1
xcopy "%SCRIPT_DIR%*.bat" "%INSTALL_DIR%\royce\" /E /I /Y /Q >nul 2>&1
xcopy "%SCRIPT_DIR%.env.example" "%INSTALL_DIR%\royce\" /I /Y /Q >nul 2>&1
echo         Files copied

:: ─── Step 3: Create venv ──────────────────────────────────────
echo.
echo  [3/8] Creating virtual environment...
echo         (This keeps Royce's stuff separate from everything else)
if not exist "%INSTALL_DIR%\royce_env" (
    python -m venv "%INSTALL_DIR%\royce_env"
    echo         Created royce_env
) else (
    echo         royce_env already exists, reusing
)

:: Activate venv
call "%INSTALL_DIR%\royce_env\Scripts\activate.bat"
echo         Activated

:: ─── Step 4: Install dependencies ─────────────────────────────
echo.
echo  [4/8] Installing dependencies...
echo         (This takes a couple minutes)
echo.
python -m pip install --upgrade pip >nul 2>&1
pip install anthropic requests beautifulsoup4 python-dotenv pygame SpeechRecognition lxml numpy sounddevice gTTS boto3 yt-dlp Pillow 2>nul
echo.
echo         Done installing

:: ─── Step 5: Create directories ───────────────────────────────
echo.
echo  [5/8] Setting up Royce's folders...
if not exist "%INSTALL_DIR%\royce\data" mkdir "%INSTALL_DIR%\royce\data"
if not exist "%INSTALL_DIR%\royce\data\memory" mkdir "%INSTALL_DIR%\royce\data\memory"
if not exist "%INSTALL_DIR%\royce\data\logs" mkdir "%INSTALL_DIR%\royce\data\logs"
if not exist "%INSTALL_DIR%\royce\data\images" mkdir "%INSTALL_DIR%\royce\data\images"
if not exist "%INSTALL_DIR%\royce\data\music_cache" mkdir "%INSTALL_DIR%\royce\data\music_cache"
echo         Folders ready

:: ─── Step 6: API Key Setup ────────────────────────────────────
echo.
echo  [6/8] Setting up API keys...
echo.
echo  ====================================================
echo   Royce needs API keys to work. You'll need at least:
echo.
echo   1. ANTHROPIC_API_KEY  (Royce's brain)
echo      Get one at: https://console.anthropic.com
echo.
echo   2. ELEVENLABS_API_KEY (Royce's voice)
echo      Get one at: https://elevenlabs.io
echo.
echo   The rest are optional but recommended:
echo   - PERPLEXITY_API_KEY  (deep research)
echo   - NEWS_API_KEY        (breaking news)
echo   - OPENAI_API_KEY      (AI images)
echo  ====================================================
echo.

:: Create .env file
set "ENV_FILE=%INSTALL_DIR%\royce\.env"
if exist "%ENV_FILE%" (
    echo  Found existing .env file. Want to reconfigure?
    set /p RECONFIG="  Type Y to reconfigure, or N to keep current keys: "
    if /i not "!RECONFIG!"=="Y" goto skip_keys
)

echo.
echo  Enter your API keys below. Press Enter to skip any.
echo.

set "ANTHROPIC_KEY="
set "ELEVENLABS_KEY="
set "ELEVENLABS_VOICE="
set "PERPLEXITY_KEY="
set "NEWS_KEY="
set "OPENAI_KEY="

set /p ANTHROPIC_KEY="  Anthropic API Key: "
set /p ELEVENLABS_KEY="  ElevenLabs API Key: "
if not "!ELEVENLABS_KEY!"=="" (
    set /p ELEVENLABS_VOICE="  ElevenLabs Voice ID (press Enter for default): "
)
set /p PERPLEXITY_KEY="  Perplexity API Key (optional): "
set /p NEWS_KEY="  NewsAPI Key (optional): "
set /p OPENAI_KEY="  OpenAI API Key (optional): "

:: Write .env file
(
echo # Royce AI Assistant — Configuration
echo.
echo # AI Brain
echo ANTHROPIC_API_KEY=!ANTHROPIC_KEY!
echo CLAUDE_MODEL=claude-sonnet-4-5-20250929
echo ROYCE_AI_PROVIDER=anthropic
echo.
echo # Voice
echo ELEVENLABS_API_KEY=!ELEVENLABS_KEY!
echo ELEVENLABS_VOICE_ID=!ELEVENLABS_VOICE!
echo ELEVENLABS_MODEL=eleven_multilingual_v2
echo.
echo # Deep Research
echo PERPLEXITY_API_KEY=!PERPLEXITY_KEY!
echo PERPLEXITY_MODEL=sonar-pro
echo.
echo # News
echo NEWS_API_KEY=!NEWS_KEY!
echo.
echo # AI Images
echo OPENAI_API_KEY=!OPENAI_KEY!
echo.
echo # Listening settings
echo MIC_ENERGY_THRESHOLD=3000
echo PAUSE_THRESHOLD=2.0
echo LISTEN_TIMEOUT=20
echo PHRASE_TIME_LIMIT=60
) > "%ENV_FILE%"

echo.
echo         API keys saved!

:skip_keys

:: ─── Step 7: Create launchers ─────────────────────────────────
echo.
echo  [7/8] Creating launch shortcuts...

:: Main launcher (double-click to run)
(
echo @echo off
echo title Royce AI Assistant
echo call "%INSTALL_DIR%\royce_env\Scripts\activate.bat"
echo cd /d "%INSTALL_DIR%"
echo python -m royce.launcher %%*
echo pause
) > "%INSTALL_DIR%\Start Royce.bat"

:: Voice mode launcher
(
echo @echo off
echo title Royce AI Assistant — Voice Mode
echo call "%INSTALL_DIR%\royce_env\Scripts\activate.bat"
echo cd /d "%INSTALL_DIR%"
echo python -m royce.launcher --mode voice
echo pause
) > "%INSTALL_DIR%\Start Royce (Voice).bat"

:: Hybrid mode launcher
(
echo @echo off
echo title Royce AI Assistant — Hybrid Mode
echo call "%INSTALL_DIR%\royce_env\Scripts\activate.bat"
echo cd /d "%INSTALL_DIR%"
echo python -m royce.launcher --mode hybrid
echo pause
) > "%INSTALL_DIR%\Start Royce (Hybrid).bat"

echo         Created launch scripts

:: ─── Step 8: Desktop shortcut ─────────────────────────────────
echo.
echo  [8/8] Creating desktop shortcut...

:: Create a VBS script to make the shortcut (only reliable way on Windows)
set "VBS_FILE=%TEMP%\create_royce_shortcut.vbs"
(
echo Set WshShell = WScript.CreateObject("WScript.Shell"^)
echo Set shortcut = WshShell.CreateShortcut(WshShell.SpecialFolders("Desktop"^) ^& "\Royce.lnk"^)
echo shortcut.TargetPath = "%INSTALL_DIR%\Start Royce.bat"
echo shortcut.WorkingDirectory = "%INSTALL_DIR%"
echo shortcut.Description = "Royce AI Assistant"
echo shortcut.Save
) > "%VBS_FILE%"

cscript //nologo "%VBS_FILE%" >nul 2>&1
del "%VBS_FILE%" >nul 2>&1

echo         Desktop shortcut created!

:: ─── Done! ────────────────────────────────────────────────────
echo.
echo  ====================================================
echo.
echo   ROYCE IS INSTALLED!
echo.
echo   How to start:
echo     - Double-click "Royce" on your desktop
echo     - Or go to %INSTALL_DIR% and pick a mode:
echo         Start Royce.bat            (text)
echo         Start Royce (Voice).bat    (voice)
echo         Start Royce (Hybrid).bat   (voice + text)
echo.
if "!ANTHROPIC_KEY!"=="" (
echo   WARNING: You didn't enter an Anthropic API key.
echo   Royce won't be able to think without it.
echo   Edit this file to add it: %INSTALL_DIR%\royce\.env
echo.
)
if "!ELEVENLABS_KEY!"=="" (
echo   NOTE: No ElevenLabs key entered.
echo   Royce will use a basic voice until you add one.
echo   Edit this file: %INSTALL_DIR%\royce\.env
echo.
)
echo   To change API keys later, edit:
echo     %INSTALL_DIR%\royce\.env
echo.
echo   Installed to: %INSTALL_DIR%
echo.
echo  ====================================================
echo.
pause
