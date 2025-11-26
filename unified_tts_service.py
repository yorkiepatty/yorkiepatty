"""
Unified TTS Service
-------------------
This module provides a unified interface for text-to-speech that automatically
selects the appropriate provider based on derek_manifest.yaml configuration.

Supported providers:
- elevenlabs (recommended)
- gtts (fallback)
- polly (AWS Polly)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedTTSService:
    """Unified TTS service that delegates to the configured provider."""

    def __init__(self):
        """Initialize the unified TTS service."""
        self.provider = "gtts"  # Default fallback
        self.voice_name = None
        self._load_manifest_config()
        self._initialize_provider()

    def _load_manifest_config(self):
        """Load TTS configuration from derek_manifest.yaml."""
        try:
            manifest_path = Path("derek_manifest.yaml")
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = yaml.safe_load(f)

                tts_config = manifest.get('tts', {})
                self.provider = tts_config.get('provider', 'gtts')
                self.voice_name = tts_config.get('voice', None)
                logger.info(f"TTS provider from manifest: {self.provider}")
                if self.voice_name:
                    logger.info(f"Default voice: {self.voice_name}")
        except Exception as e:
            logger.warning(f"Could not load manifest, using default provider: {e}")

    def _initialize_provider(self):
        """Initialize the selected TTS provider."""
        if self.provider == "elevenlabs":
            try:
                from elevenlabs_tts_service import get_elevenlabs_service
                self.service = get_elevenlabs_service()
                if self.service.is_available():
                    logger.info("Using ElevenLabs TTS service")
                    self.text_to_speech_impl = self._elevenlabs_tts
                else:
                    logger.warning("ElevenLabs not available, falling back to gTTS")
                    self.provider = "gtts"
                    self._initialize_gtts()
            except ImportError as e:
                logger.warning(f"ElevenLabs import failed, falling back to gTTS: {e}")
                self.provider = "gtts"
                self._initialize_gtts()

        elif self.provider == "polly":
            try:
                import boto3
                self.polly_client = boto3.client('polly', region_name=os.getenv('AWS_REGION', 'us-east-1'))
                self.text_to_speech_impl = self._polly_tts
                logger.info("Using AWS Polly TTS service")
            except Exception as e:
                logger.warning(f"AWS Polly initialization failed, falling back to gTTS: {e}")
                self.provider = "gtts"
                self._initialize_gtts()

        else:  # gtts or fallback
            self._initialize_gtts()

    def _initialize_gtts(self):
        """Initialize gTTS as the provider."""
        try:
            from gtts import gTTS
            self.text_to_speech_impl = self._gtts_tts
            logger.info("Using gTTS service")
        except ImportError:
            logger.error("gTTS not available. Please install: pip install gTTS")
            self.text_to_speech_impl = None

    def _elevenlabs_tts(self, text: str, voice: Optional[str] = None, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """Use ElevenLabs for TTS.

        Note: If voice is None, the ElevenLabs service will use its own configured
        voice ID (prioritizing ELEVENLABS_VOICE_ID from .env, then manifest, then default).
        """
        # Don't override with manifest voice - let ElevenLabs service handle its own defaults
        return self.service.text_to_speech(text, voice=voice, output_path=output_path, **kwargs)

    def _gtts_tts(self, text: str, voice_id: str = "calming", output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """Use gTTS for TTS."""
        try:
            from advanced_tts_service import text_to_speech as gtts_text_to_speech
            return gtts_text_to_speech(text, voice_id=voice_id, output_path=output_path)
        except ImportError:
            # Fallback to simple gTTS
            from gtts import gTTS
            import tempfile
            import hashlib

            if not output_path:
                cache_dir = Path("static/audio")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_key = hashlib.md5(text.encode()).hexdigest()
                output_path = str(cache_dir / f"{cache_key}.mp3")

            if os.path.exists(output_path):
                return output_path

            tts = gTTS(text=text, lang='en')
            tts.save(output_path)
            return output_path

    def _polly_tts(self, text: str, voice: Optional[str] = None, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """Use AWS Polly for TTS."""
        import hashlib
        from pathlib import Path

        voice = voice or os.getenv('POLLY_VOICE_ID', 'Joanna')
        engine = os.getenv('POLLY_ENGINE', 'neural')

        if not output_path:
            cache_dir = Path("static/audio")
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = hashlib.md5(f"{text}|{voice}".encode()).hexdigest()
            output_path = str(cache_dir / f"{cache_key}.mp3")

        if os.path.exists(output_path):
            return output_path

        try:
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice,
                Engine=engine
            )

            with open(output_path, 'wb') as f:
                f.write(response['AudioStream'].read())

            return output_path
        except Exception as e:
            logger.error(f"Polly TTS failed: {e}")
            return None

    def text_to_speech(self, text: str, voice: Optional[str] = None, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Convert text to speech using the configured provider.

        Args:
            text: Text to convert to speech
            voice: Voice name/ID (provider-specific)
            output_path: Optional path to save audio file
            **kwargs: Additional provider-specific parameters

        Returns:
            Path to generated audio file, or None if failed
        """
        if not self.text_to_speech_impl:
            logger.error("No TTS provider available")
            return None

        try:
            return self.text_to_speech_impl(text, voice=voice, output_path=output_path, **kwargs)
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

    def get_provider_info(self):
        """Get information about the current TTS provider."""
        return {
            "provider": self.provider,
            "default_voice": self.voice_name,
            "available": self.text_to_speech_impl is not None
        }


# Create singleton instance
_unified_service = None


def get_tts_service() -> UnifiedTTSService:
    """Get or create the unified TTS service singleton."""
    global _unified_service
    if _unified_service is None:
        _unified_service = UnifiedTTSService()
    return _unified_service


def text_to_speech(text: str, voice: Optional[str] = None, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
    """
    Convenience function for text-to-speech conversion.

    Args:
        text: Text to convert
        voice: Voice name/ID (optional)
        output_path: Output file path (optional)
        **kwargs: Additional parameters for the provider

    Returns:
        Path to generated audio file, or None if failed
    """
    service = get_tts_service()
    return service.text_to_speech(text, voice, output_path, **kwargs)


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?"
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
