"""
ElevenLabs TTS Service
----------------------
This module provides text-to-speech capabilities using ElevenLabs API.
It reads configuration from derek_manifest.yaml and .env
"""

import hashlib
import logging
import os
from typing import Optional
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ElevenLabs
try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("ElevenLabs package not installed. Install with: pip install elevenlabs")


class ElevenLabsTTSService:
    """Text-to-Speech service using ElevenLabs API."""

    def __init__(self):
        """Initialize the ElevenLabs TTS service."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = None
        self.voice_id = None  # Voice ID (priority: .env > manifest > default)
        self.voice_name = "Sunny"  # Default voice name
        self.cache_dir = Path("static/audio")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration from manifest
        self._load_manifest_config()

        # Override with environment variable if set
        env_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        if env_voice_id and env_voice_id != "your_elevenlabs_voice_id_here":
            self.voice_id = env_voice_id
            logger.info(f"Using ElevenLabs voice ID from .env: {self.voice_id}")
        elif self.voice_id:
            logger.info(f"Using ElevenLabs voice ID from manifest: {self.voice_id}")
        else:
            # Use voice name as fallback
            self.voice_id = self.voice_name
            logger.info(f"Using voice name as ID: {self.voice_id}")

        # Initialize client if API key is available
        if self.api_key and self.api_key != "your_elevenlabs_api_key_here":
            if ELEVENLABS_AVAILABLE:
                try:
                    self.client = ElevenLabs(api_key=self.api_key)
                    logger.info("ElevenLabs TTS service initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize ElevenLabs client: {e}")
            else:
                logger.error("ElevenLabs package not available")
        else:
            logger.warning("ELEVENLABS_API_KEY not configured in .env file")

    def _load_manifest_config(self):
        """Load configuration from derek_manifest.yaml."""
        try:
            manifest_path = Path("derek_manifest.yaml")
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = yaml.safe_load(f)

                # Load TTS settings
                tts_config = manifest.get('tts', {})
                if tts_config.get('provider') == 'elevenlabs':
                    voice = tts_config.get('voice', 'Sunny')
                    # Voice can be either a name or ID in the manifest
                    self.voice_name = voice
                    self.voice_id = voice  # May be overridden by .env
                    logger.info(f"Loaded ElevenLabs voice from manifest: {voice}")
        except Exception as e:
            logger.warning(f"Could not load manifest configuration: {e}")

    def is_available(self) -> bool:
        """Check if ElevenLabs service is available."""
        return self.client is not None and ELEVENLABS_AVAILABLE

    def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
        model: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True
    ) -> Optional[str]:
        """
        Convert text to speech using ElevenLabs.

        Args:
            text: Text to convert to speech
            voice: Voice ID or name (defaults to ELEVENLABS_VOICE_ID from .env or manifest voice)
            output_path: Optional path to save audio file
            model: ElevenLabs model to use
            stability: Voice stability (0.0-1.0)
            similarity_boost: Voice similarity boost (0.0-1.0)
            style: Style exaggeration (0.0-1.0)
            use_speaker_boost: Enable speaker boost

        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.is_available():
            logger.error("ElevenLabs service not available. Check API key and installation.")
            return None

        voice = voice or self.voice_id

        try:
            # Generate cache key
            cache_key = hashlib.md5(
                f"{text}|{voice}|{model}|{stability}|{similarity_boost}".encode()
            ).hexdigest()

            # Determine output path
            if not output_path:
                output_path = str(self.cache_dir / f"{cache_key}.mp3")

            # Return cached file if it exists
            if os.path.exists(output_path):
                logger.info(f"Using cached audio: {output_path}")
                return output_path

            # Generate speech
            logger.info(f"Generating speech with ElevenLabs voice: {voice}")

            response = self.client.text_to_speech.convert(
                voice_id=voice,
                output_format="mp3_44100_128",
                text=text,
                model_id=model,
                voice_settings=VoiceSettings(
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=style,
                    use_speaker_boost=use_speaker_boost,
                ),
            )

            # Save audio file
            with open(output_path, 'wb') as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)

            logger.info(f"Audio saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ElevenLabs TTS generation failed: {e}")
            return None

    def get_available_voices(self):
        """Get list of available voices from ElevenLabs."""
        if not self.is_available():
            return []

        try:
            response = self.client.voices.get_all()
            voices = []
            for voice in response.voices:
                voices.append({
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category if hasattr(voice, 'category') else None,
                })
            return voices
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []


# Create singleton instance
_elevenlabs_service = None


def get_elevenlabs_service() -> ElevenLabsTTSService:
    """Get or create the ElevenLabs TTS service singleton."""
    global _elevenlabs_service
    if _elevenlabs_service is None:
        _elevenlabs_service = ElevenLabsTTSService()
    return _elevenlabs_service


def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Convenience function for text-to-speech conversion.

    Args:
        text: Text to convert
        voice: Voice name (optional)
        output_path: Output file path (optional)
        **kwargs: Additional parameters for ElevenLabs

    Returns:
        Path to generated audio file, or None if failed
    """
    service = get_elevenlabs_service()
    return service.text_to_speech(text, voice, output_path, **kwargs)


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?"
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
