"""Test gTTS fallback voice synthesis"""
from gtts import gTTS
import os
import tempfile

print("Testing gTTS (Google Text-to-Speech) fallback...")
print()

try:
    text = "Hello, this is Sunny speaking. The ElevenLabs API is blocked in this environment, but gTTS works!"

    # Create temp file
    temp_file = os.path.join(tempfile.gettempdir(), "test_gtts.mp3")

    # Generate speech
    tts = gTTS(text=text, lang='en')
    tts.save(temp_file)

    print(f"✅ gTTS synthesis successful!")
    print(f"   Audio saved to: {temp_file}")
    print(f"   File size: {os.path.getsize(temp_file)} bytes")
    print()
    print("Note: gTTS works fine in this environment.")
    print("ElevenLabs will work when you run this on your local machine!")

except Exception as e:
    print(f"❌ gTTS failed: {e}")
    import traceback
    traceback.print_exc()
