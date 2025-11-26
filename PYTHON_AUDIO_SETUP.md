# Python Audio Setup Reference

## Python Version
- **Current Python**: 3.11.14 ✅
- **Compatible**: Python 3.8+ (recommended 3.11+)

## PyAudio Dependency

### Files that require PyAudio:
1. **`services/speech_recognition_engine.py:123`**
   - Uses `sr.Microphone()` for live microphone input
   - Required for real-time speech recognition

2. **`sunny_ultimate_voice.py:122`**
   - Ultimate voice system with speech recognition
   - Optional: can disable with `enable_speech=False`

3. **`derek_ultimate_voice.py:122`**
   - Derek voice system with speech recognition
   - Optional: can disable with `enable_speech=False`

### Why PyAudio?
The `SpeechRecognition` library (in requirements.txt) uses PyAudio as a backend for microphone access. While SpeechRecognition is listed in requirements.txt, PyAudio must be explicitly installed.

## Installation

### Install all dependencies including PyAudio:
```bash
pip install -r requirements.txt
```

### Install PyAudio separately (if needed):
```bash
pip install PyAudio==0.2.14
```

### Platform-specific notes:

**macOS:**
```bash
# May need portaudio first
brew install portaudio
pip install pyaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**Windows:**
```bash
# Usually works directly
pip install pyaudio
```

## Alternative: Disable Speech Recognition

If you don't need microphone input, you can disable speech recognition:

```python
# In sunny_ultimate_voice.py or derek_ultimate_voice.py
voice_system = SunnyUltimateVoice(enable_speech=False)
```

## Other Audio Libraries in Use
- `sounddevice==0.5.3` - Audio I/O
- `soundfile==0.13.1` - Audio file reading/writing
- `SpeechRecognition==3.14.3` - Speech recognition (uses PyAudio)
- `vosk==0.3.44` - Offline speech recognition
- `gTTS==2.5.4` - Google Text-to-Speech
- `librosa==0.11.0` - Audio analysis
- `pygame==2.6.1` - Audio playback
