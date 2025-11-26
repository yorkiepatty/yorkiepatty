# ElevenLabs TTS Integration

This project now supports ElevenLabs for high-quality text-to-speech, as configured in `derek_manifest.yaml`.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the `elevenlabs` package (version 1.10.0) along with other dependencies.

### 2. Get Your ElevenLabs API Key

1. Go to [ElevenLabs](https://elevenlabs.io/)
2. Sign up or log in to your account
3. Navigate to your profile settings
4. Copy your API key

### 3. Configure Your API Key

Edit your `.env` file and replace the placeholder:

```bash
# ElevenLabs Voice Settings (recommended - see derek_manifest.yaml)
ELEVENLABS_API_KEY=your_actual_api_key_here
```

### 4. Verify Configuration

Run the test script to verify everything is working:

```bash
python test_elevenlabs_tts.py
```

This will:
- Check your configuration
- Test the ElevenLabs connection
- List available voices
- Generate a test audio file

## Usage

### Using the Unified TTS Service (Recommended)

The unified service automatically selects the provider based on `derek_manifest.yaml`:

```python
from unified_tts_service import text_to_speech

# Generate speech (uses ElevenLabs automatically)
audio_path = text_to_speech("Hello, this is Derek speaking!")

# Use a specific voice
audio_path = text_to_speech("Hello!", voice="Derek")

# Save to a specific path
audio_path = text_to_speech("Hello!", output_path="my_audio.mp3")
```

### Using ElevenLabs Directly

If you want to use ElevenLabs directly with more control:

```python
from elevenlabs_tts_service import get_elevenlabs_service

service = get_elevenlabs_service()

# Check if available
if service.is_available():
    # Generate speech with custom parameters
    audio_path = service.text_to_speech(
        text="Hello, world!",
        voice="Derek",
        stability=0.5,
        similarity_boost=0.75,
        style=0.0
    )
```

### Getting Available Voices

```python
from elevenlabs_tts_service import get_elevenlabs_service

service = get_elevenlabs_service()
voices = service.get_available_voices()

for voice in voices:
    print(f"{voice['name']} (ID: {voice['voice_id']})")
```

## Configuration

### derek_manifest.yaml

The TTS provider is configured in `derek_manifest.yaml`:

```yaml
tts:
  provider: "elevenlabs"  # Can be: elevenlabs, gtts, polly
  voice: "Derek"          # Default voice name
  enable_auto_speech: true
```

### Voice Settings

You can customize ElevenLabs voice parameters:

- **stability** (0.0-1.0): Higher values make speech more consistent
- **similarity_boost** (0.0-1.0): How closely to match the voice
- **style** (0.0-1.0): Style exaggeration
- **use_speaker_boost** (bool): Enhance speaker clarity

## Fallback Behavior

If ElevenLabs is not available (missing API key, network issues, etc.), the system automatically falls back to:

1. **gTTS** (Google Text-to-Speech) - Free, no API key required
2. **AWS Polly** - If AWS credentials are configured

## File Structure

- `elevenlabs_tts_service.py` - ElevenLabs-specific implementation
- `unified_tts_service.py` - Provider-agnostic interface
- `test_elevenlabs_tts.py` - Test and verification script
- `advanced_tts_service.py` - Legacy gTTS service (still available)
- `tts_service.py` - Legacy gTTS service (still available)

## Migrating Existing Code

If you have existing code using the old TTS services, you can migrate easily:

**Old:**
```python
from advanced_tts_service import text_to_speech
audio = text_to_speech("Hello", voice_id="calming")
```

**New:**
```python
from unified_tts_service import text_to_speech
audio = text_to_speech("Hello", voice="Derek")
```

The unified service will automatically use ElevenLabs based on your manifest configuration!

## Troubleshooting

### "ElevenLabs service not available"

**Possible causes:**
- API key not set in `.env`
- API key is still the placeholder value
- `elevenlabs` package not installed

**Solution:**
```bash
pip install elevenlabs
```
Then update your `.env` file with a valid API key.

### "No voices retrieved"

**Possible causes:**
- Invalid API key
- Network connectivity issues
- API rate limits exceeded

**Solution:**
- Verify your API key at elevenlabs.io
- Check your internet connection
- Wait a few minutes if rate-limited

### Audio not playing

The TTS service generates audio files but doesn't play them automatically. To play audio:

```python
import pygame

pygame.mixer.init()
pygame.mixer.music.load(audio_path)
pygame.mixer.music.play()
```

Or use your system's audio player.

## Cost Considerations

ElevenLabs is a paid service with a free tier. Check your usage at:
- [ElevenLabs Pricing](https://elevenlabs.io/pricing)

The unified service includes caching to avoid regenerating the same audio, which helps reduce API calls and costs.

## Support

For issues specific to:
- **ElevenLabs API**: Visit [ElevenLabs Documentation](https://elevenlabs.io/docs)
- **This integration**: Check the test output from `test_elevenlabs_tts.py`

---

**© 2025 Everett Nathaniel Christman & Misty Gail Christman**
The Christman AI Project — Luma Cognify AI
