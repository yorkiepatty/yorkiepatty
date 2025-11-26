#!/usr/bin/env python3
"""
Test script for ElevenLabs TTS integration.

This script tests the unified TTS service to ensure ElevenLabs is properly configured.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_unified_tts_service():
    """Test the unified TTS service."""
    print("\n" + "="*70)
    print("Testing Unified TTS Service")
    print("="*70 + "\n")

    try:
        from unified_tts_service import get_tts_service

        # Get the service
        service = get_tts_service()

        # Show provider info
        info = service.get_provider_info()
        print(f"Provider: {info['provider']}")
        print(f"Default Voice: {info['default_voice']}")
        print(f"Available: {info['available']}")
        print()

        if not info['available']:
            print("❌ TTS service not available")
            return False

        # Test text-to-speech generation
        test_text = "Hello! This is Sunny speaking. Testing the ElevenLabs text-to-speech integration."
        print(f"Generating speech: '{test_text}'")
        print()

        audio_path = service.text_to_speech(test_text)

        if audio_path and Path(audio_path).exists():
            print(f"✅ Audio generated successfully: {audio_path}")
            print(f"   File size: {Path(audio_path).stat().st_size} bytes")
            return True
        else:
            print("❌ Audio generation failed")
            return False

    except Exception as e:
        print(f"❌ Error testing TTS service: {e}")
        logger.exception("Test failed with exception")
        return False


def test_elevenlabs_direct():
    """Test ElevenLabs service directly."""
    print("\n" + "="*70)
    print("Testing ElevenLabs Service Directly")
    print("="*70 + "\n")

    try:
        from elevenlabs_tts_service import get_elevenlabs_service

        service = get_elevenlabs_service()

        if not service.is_available():
            print("❌ ElevenLabs service not available")
            print("   Possible reasons:")
            print("   - ELEVENLABS_API_KEY not set in .env")
            print("   - API key is placeholder value")
            print("   - elevenlabs package not installed")
            print()
            print("To fix:")
            print("1. Install elevenlabs: pip install elevenlabs")
            print("2. Get API key from: https://elevenlabs.io/")
            print("3. Update ELEVENLABS_API_KEY in .env file")
            return False

        print("✅ ElevenLabs service is available")
        print(f"   Voice: {service.voice_name}")
        print()

        # Try to get available voices
        print("Fetching available voices...")
        voices = service.get_available_voices()

        if voices:
            print(f"✅ Found {len(voices)} available voices:")
            for voice in voices[:5]:  # Show first 5
                print(f"   - {voice['name']} (ID: {voice['voice_id']})")
            if len(voices) > 5:
                print(f"   ... and {len(voices) - 5} more")
        else:
            print("⚠️  No voices retrieved (API key may be invalid)")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install with: pip install elevenlabs")
        return False
    except Exception as e:
        print(f"❌ Error testing ElevenLabs: {e}")
        logger.exception("ElevenLabs test failed")
        return False


def check_configuration():
    """Check configuration files."""
    print("\n" + "="*70)
    print("Checking Configuration")
    print("="*70 + "\n")

    import os
    from dotenv import load_dotenv
    import yaml

    # Load .env
    load_dotenv()

    # Check manifest
    manifest_path = Path("derek_manifest.yaml")
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        tts_config = manifest.get('tts', {})
        print(f"✅ Manifest found")
        print(f"   Provider: {tts_config.get('provider', 'not set')}")
        print(f"   Voice: {tts_config.get('voice', 'not set')}")
    else:
        print("❌ derek_manifest.yaml not found")

    print()

    # Check API key
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if api_key and api_key != 'your_elevenlabs_api_key_here':
        print(f"✅ ELEVENLABS_API_KEY is set")
        print(f"   Length: {len(api_key)} characters")
    else:
        print("❌ ELEVENLABS_API_KEY not configured")
        print("   Update your .env file with a valid API key")

    print()

    # Check elevenlabs package
    try:
        import elevenlabs
        print(f"✅ elevenlabs package installed")
        print(f"   Version: {elevenlabs.__version__ if hasattr(elevenlabs, '__version__') else 'unknown'}")
    except ImportError:
        print("❌ elevenlabs package not installed")
        print("   Install with: pip install elevenlabs")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ElevenLabs TTS Integration Test")
    print("="*70)

    # Check configuration
    check_configuration()

    # Test ElevenLabs directly
    elevenlabs_ok = test_elevenlabs_direct()

    # Test unified service
    unified_ok = test_unified_tts_service()

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"ElevenLabs Direct: {'✅ PASS' if elevenlabs_ok else '❌ FAIL'}")
    print(f"Unified Service: {'✅ PASS' if unified_ok else '❌ FAIL'}")
    print("="*70 + "\n")

    if elevenlabs_ok and unified_ok:
        print("🎉 All tests passed! ElevenLabs is ready to use.")
        print("\nNext steps:")
        print("1. Make sure your ELEVENLABS_API_KEY is valid")
        print("2. The system will automatically use ElevenLabs based on derek_manifest.yaml")
        print("3. Import with: from unified_tts_service import text_to_speech")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    main()


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?"
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
