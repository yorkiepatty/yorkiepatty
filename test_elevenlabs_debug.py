"""
Debug ElevenLabs API connection
"""
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")
voice_id = os.getenv("ELEVENLABS_VOICE_ID")

print("=" * 60)
print("ElevenLabs API Debug")
print("=" * 60)
print(f"API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'NOT SET'}")
print(f"Voice ID: {voice_id}")
print()

try:
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)

    print("Testing API connection...")
    print()

    # Try to get user info
    try:
        user = client.user.get()
        print("✅ API Key is valid!")
        print(f"   User: {user}")
        print()
    except Exception as e:
        print(f"❌ API Key validation failed: {e}")
        print()

    # Try to get voices
    try:
        print("Fetching available voices...")
        response = client.voices.get_all()
        print(f"✅ Found {len(response.voices)} voices:")
        for voice in response.voices[:5]:  # Show first 5
            print(f"   - {voice.name}: {voice.voice_id}")
        print()
    except Exception as e:
        print(f"❌ Failed to get voices: {e}")
        print()

    # Try to generate a test audio
    try:
        print(f"Testing speech generation with voice ID: {voice_id}")
        response = client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="mp3_44100_128",
            text="Hello, this is a test.",
            model_id="eleven_multilingual_v2"
        )

        # Save to test file
        with open("/tmp/test_elevenlabs.mp3", "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print("✅ Speech generation successful!")
        print("   Saved to: /tmp/test_elevenlabs.mp3")

    except Exception as e:
        print(f"❌ Speech generation failed: {e}")
        import traceback
        traceback.print_exc()

except ImportError:
    print("❌ elevenlabs package not installed")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
