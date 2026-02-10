"""
Royce Voice System
------------------
Text-to-speech output and speech recognition input.
Supports ElevenLabs (primary), AWS Polly, and gTTS as fallbacks.
Patient listening with generous pause detection.
"""

import os
import io
import re
import time
import uuid
import logging
import tempfile
import threading
from typing import Optional, Dict, Callable

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Try importing speech/audio libraries
try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class RoyceVoice:
    """Handles all voice I/O for Royce — speaking and listening.

    TTS priority order:
    1. ElevenLabs (best quality, most natural)
    2. AWS Polly (neural)
    3. gTTS (free fallback)
    """

    def __init__(self, voice_id: str = "Matthew", engine: str = "neural"):
        self.voice_id = voice_id
        self.polly_engine = engine
        self.is_speaking = False
        self.is_listening = False
        self._stop_listening = False

        # ─── ElevenLabs Config ─────────────────────────────────────
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")
        self.elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
        self.elevenlabs_url = "https://api.elevenlabs.io/v1"

        if self.elevenlabs_api_key:
            logger.info(f"ElevenLabs initialized (voice: {self.elevenlabs_voice_id or 'default'})")

        # ─── AWS Polly Config ──────────────────────────────────────
        self.polly = None
        if BOTO3_AVAILABLE and not self.elevenlabs_api_key:
            try:
                self.polly = boto3.client("polly", region_name=os.getenv("AWS_REGION", "us-east-1"))
                logger.info(f"AWS Polly initialized (voice: {voice_id})")
            except Exception as e:
                logger.warning(f"AWS Polly not available: {e}")

        # ─── Speech Recognition Config ─────────────────────────────
        self.recognizer = None
        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
            # Patient listening settings — give people TIME
            self.recognizer.energy_threshold = int(os.getenv("MIC_ENERGY_THRESHOLD", "3000"))
            self.recognizer.pause_threshold = float(os.getenv("PAUSE_THRESHOLD", "2.0"))
            self.recognizer.phrase_threshold = 0.3
            self.recognizer.non_speaking_duration = 1.0
            logger.info("Speech recognition initialized (patient mode)")

        logger.info("Royce Voice System ready")

    # ─── Speaking ──────────────────────────────────────────────────

    def speak(self, text: str):
        """Speak the given text aloud.

        Priority: ElevenLabs > AWS Polly > gTTS > log only
        """
        if not text or not text.strip():
            return

        # Clean text before speaking
        text = self._clean_for_speech(text)

        self.is_speaking = True

        if self.elevenlabs_api_key and PYGAME_AVAILABLE:
            self._speak_elevenlabs(text)
        elif self.polly and PYGAME_AVAILABLE:
            self._speak_polly(text)
        elif GTTS_AVAILABLE and PYGAME_AVAILABLE:
            self._speak_gtts(text)
        else:
            logger.info(f"[Royce says]: {text}")

        self.is_speaking = False

    # ─── ElevenLabs TTS ────────────────────────────────────────────

    def _speak_elevenlabs(self, text: str):
        """Speak using ElevenLabs API — highest quality voice synthesis."""
        try:
            chunks = self._split_text(text, max_length=4500)

            for chunk in chunks:
                voice_id = self.elevenlabs_voice_id or self._get_default_elevenlabs_voice()
                url = f"{self.elevenlabs_url}/text-to-speech/{voice_id}"

                headers = {
                    "xi-api-key": self.elevenlabs_api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                }

                payload = {
                    "text": chunk,
                    "model_id": self.elevenlabs_model,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.3,
                        "use_speaker_boost": True,
                    },
                }

                response = requests.post(url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()

                self._play_audio_bytes(response.content)

        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            # Fall through to Polly
            if self.polly:
                self._speak_polly(text)
            elif GTTS_AVAILABLE:
                self._speak_gtts(text)

    def _get_default_elevenlabs_voice(self) -> str:
        """Get the first available ElevenLabs voice if none configured."""
        try:
            headers = {"xi-api-key": self.elevenlabs_api_key}
            response = requests.get(f"{self.elevenlabs_url}/voices", headers=headers, timeout=10)
            response.raise_for_status()
            voices = response.json().get("voices", [])
            if voices:
                voice_id = voices[0]["voice_id"]
                self.elevenlabs_voice_id = voice_id
                logger.info(f"Using ElevenLabs voice: {voices[0].get('name', voice_id)}")
                return voice_id
        except Exception as e:
            logger.error(f"Failed to fetch ElevenLabs voices: {e}")
        return ""

    def list_elevenlabs_voices(self) -> list:
        """List all available ElevenLabs voices."""
        if not self.elevenlabs_api_key:
            return []

        try:
            headers = {"xi-api-key": self.elevenlabs_api_key}
            response = requests.get(f"{self.elevenlabs_url}/voices", headers=headers, timeout=10)
            response.raise_for_status()
            voices = response.json().get("voices", [])
            return [
                {
                    "voice_id": v["voice_id"],
                    "name": v.get("name", "Unknown"),
                    "category": v.get("category", ""),
                    "description": v.get("description", ""),
                }
                for v in voices
            ]
        except Exception as e:
            logger.error(f"Failed to list ElevenLabs voices: {e}")
            return []

    def set_elevenlabs_voice(self, voice_id: str):
        """Switch to a different ElevenLabs voice."""
        self.elevenlabs_voice_id = voice_id
        logger.info(f"ElevenLabs voice set to: {voice_id}")

    # ─── AWS Polly TTS ─────────────────────────────────────────────

    def _speak_polly(self, text: str):
        """Speak using AWS Polly neural voices."""
        try:
            chunks = self._split_text(text, max_length=2500)

            for chunk in chunks:
                response = self.polly.synthesize_speech(
                    Text=chunk,
                    OutputFormat="mp3",
                    VoiceId=self.voice_id,
                    Engine=self.polly_engine,
                    SampleRate="22050",
                )

                audio_data = response["AudioStream"].read()
                self._play_audio_bytes(audio_data)

        except Exception as e:
            logger.error(f"Polly TTS error: {e}")
            if GTTS_AVAILABLE:
                self._speak_gtts(text)

    # ─── gTTS Fallback ─────────────────────────────────────────────

    def _speak_gtts(self, text: str):
        """Fallback: speak using Google TTS."""
        try:
            tts = gTTS(text=text, lang="en")
            temp_file = os.path.join(tempfile.gettempdir(), f"royce_{uuid.uuid4()}.mp3")
            tts.save(temp_file)

            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            os.remove(temp_file)
        except Exception as e:
            logger.error(f"gTTS error: {e}")

    # ─── Audio Playback ────────────────────────────────────────────

    def _play_audio_bytes(self, audio_bytes: bytes):
        """Play raw audio bytes through pygame."""
        try:
            audio_stream = io.BytesIO(audio_bytes)
            pygame.mixer.music.load(audio_stream, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    # ─── Listening (Patient Mode) ──────────────────────────────────

    def listen(self, timeout: int = 20, phrase_limit: int = 60) -> Optional[str]:
        """Listen for speech input with patient pause detection.

        Royce gives plenty of time for the user to say what they need.
        Uses a generous pause threshold so people can think mid-sentence.

        Args:
            timeout: Max seconds to wait for speech to begin
            phrase_limit: Max seconds for a single phrase

        Returns:
            Transcribed text or None if nothing detected
        """
        if not SR_AVAILABLE or not self.recognizer:
            logger.warning("Speech recognition not available")
            return None

        try:
            with sr.Microphone() as source:
                self.is_listening = True
                logger.info("Listening... (take your time)")

                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen with generous timeouts
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )

                self.is_listening = False
                logger.info("Processing what you said...")

                # Transcribe
                text = self.recognizer.recognize_google(audio)
                return text.strip()

        except sr.WaitTimeoutError:
            self.is_listening = False
            return None
        except sr.UnknownValueError:
            self.is_listening = False
            logger.debug("Couldn't make out what was said")
            return None
        except sr.RequestError as e:
            self.is_listening = False
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            self.is_listening = False
            logger.error(f"Listening error: {e}")
            return None

    def listen_continuous(self, callback: Callable[[str], None]):
        """Continuously listen and call back with each recognized phrase.

        Designed for hands-free conversation mode.
        """
        self._stop_listening = False

        def _loop():
            while not self._stop_listening:
                text = self.listen()
                if text:
                    callback(text)
                time.sleep(0.3)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()
        return thread

    def stop_listening(self):
        """Stop continuous listening."""
        self._stop_listening = True
        self.is_listening = False

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Clean text for natural-sounding speech output."""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Clean up whitespace
        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _split_text(text: str, max_length: int = 2500) -> list:
        """Split long text into speakable chunks at sentence boundaries."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= max_length:
                current += (" " + sentence if current else sentence)
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks
