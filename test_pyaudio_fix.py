#!/usr/bin/env python3
"""
Test script to verify that sunny_ultimate_voice.py works without PyAudio
"""

print("Testing Sunny Ultimate Voice without PyAudio...")
print("=" * 60)

try:
    # Try importing speech_recognition
    import speech_recognition as sr
    print("✅ speech_recognition module imported successfully")

    # Try to create a Microphone object (this is where PyAudio is required)
    try:
        mic = sr.Microphone()
        print("✅ PyAudio is installed and working!")
    except AttributeError as e:
        print("⚠️  PyAudio not installed (expected on Windows)")
        print(f"   Error: {e}")
        print("   This is OK - the application should handle this gracefully")

except ImportError as e:
    print(f"❌ Could not import speech_recognition: {e}")
    print("   Run: pip install SpeechRecognition")

print("\n" + "=" * 60)
print("Now testing Sunny initialization...")
print("=" * 60)

try:
    from sunny_ultimate_voice import SunnyUltimateVoice
    print("✅ SunnyUltimateVoice imported successfully")

    # This should work even without PyAudio
    print("\nTrying to initialize Sunny (this should work without PyAudio)...")
    sunny = SunnyUltimateVoice(
        ai_provider="auto",
        voice_id="matthew",
        use_web_search=True,
        enable_speech=True  # This should gracefully degrade if PyAudio is missing
    )

    print("\n✅ SUCCESS! Sunny initialized without crashing!")
    print(f"   Speech enabled: {sunny.enable_speech}")
    print(f"   AI Provider: {sunny.ai_provider}")

    if not sunny.enable_speech:
        print("\n✅ Speech recognition properly disabled (PyAudio not available)")
        print("   Sunny will use text input instead")
    else:
        print("\n✅ Speech recognition enabled and working!")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
