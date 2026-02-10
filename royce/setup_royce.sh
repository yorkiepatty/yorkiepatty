#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Royce AI Assistant — Linux/Mac Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo ""
echo "===================================================="
echo "  ROYCE AI ASSISTANT — Setup"
echo "  Confident. Smart. Caring."
echo "===================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo "Install Python 3.9+ from your package manager or https://www.python.org"
    exit 1
fi

echo "[1/5] Upgrading pip..."
python3 -m pip install --upgrade pip

echo ""
echo "[2/5] Installing core dependencies..."
pip3 install anthropic requests beautifulsoup4 python-dotenv pygame SpeechRecognition lxml

echo ""
echo "[3/5] Installing voice/audio dependencies..."
pip3 install gTTS boto3
pip3 install PyAudio 2>/dev/null || echo "[NOTE] PyAudio install failed. Try: sudo apt install portaudio19-dev && pip3 install PyAudio"

echo ""
echo "[4/5] Installing optional dependencies..."
pip3 install yt-dlp 2>/dev/null
pip3 install Pillow 2>/dev/null

echo ""
echo "[5/5] Setting up directories..."
mkdir -p data/memory data/logs data/images data/music_cache

# Check for .env
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo ""
    echo "[IMPORTANT] Created .env from .env.example"
    echo "Edit .env and add your API keys before running Royce."
fi

echo ""
echo "===================================================="
echo "  Setup Complete!"
echo ""
echo "  To start Royce:"
echo "    python3 -m royce.launcher               (text mode)"
echo "    python3 -m royce.launcher --mode voice   (voice mode)"
echo "    python3 -m royce.launcher --mode hybrid  (voice + text)"
echo ""
echo "  Don't forget to edit .env with your API keys!"
echo "===================================================="
echo ""
