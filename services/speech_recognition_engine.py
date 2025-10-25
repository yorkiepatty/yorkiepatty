import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
import speech_recognition as sr

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AUDIO_CACHE_DIR = "audio_cache"
if not os.path.exists(AUDIO_CACHE_DIR):
    os.makedirs(AUDIO_CACHE_DIR)


class SpeechRecognitionEngine:
    def __init__(
        self, language: str = "en-US", simulate: bool = False, device_index: int = None
    ):
        self.language = language
        self.simulate = simulate
        self.device_index = device_index
        self.is_listening = False
        self.callbacks = []
        self.recognizer = sr.Recognizer()
        logger.info(
            f"Speech Engine init: lang={language}, simulate={simulate}, device={device_index}"
        )

    def start_listening(self, callback: Optional[Callable] = None) -> bool:
        if self.is_listening:
            logger.warning("Speech recognition is already active")
            return False
        if callback:
            self.callbacks.append(callback)
        self.is_listening = True
        self._start_listening_thread()
        return True

    def stop_listening(self) -> bool:
        if not self.is_listening:
            logger.warning("Speech recognition is not active")
            return False
        self.is_listening = False
        return True

    def _start_listening_thread(self):
        thread = threading.Thread(target=self._audio_processing_loop)
        thread.daemon = True
        thread.start()

    def recognize_from_bytes(
        self, audio_bytes: bytes, sample_rate: int = 16000
    ) -> Tuple[str, float, Dict[str, Any]]:
        audio_data = sr.AudioData(audio_bytes, sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio_data, language=self.language)
            return (
                text,
                0.9,
                {
                    "language": self.language,
                    "duration": len(audio_bytes) / (sample_rate * 2),
                    "timestamp": time.time(),
                },
            )
        except sr.UnknownValueError:
            return "❌ [Unrecognized speech]", 0.0, {"error": "unrecognized"}
        except sr.RequestError as e:
            return "❌ [Speech API error]", 0.0, {"error": str(e)}

    def _audio_processing_loop(self):
        if self.simulate:
            self._simulate_loop()
        elif self.device_index == -1:
            self._file_audio_loop()
        else:
            self._microphone_loop()

    def _simulate_loop(self):
        phrases = [
            "Hello, how are you?",
            "What can you help me with?",
            "I need assistance.",
        ]
        while self.is_listening:
            text = phrases[int(time.time()) % len(phrases)]
            for cb in self.callbacks:
                cb(
                    text,
                    0.95,
                    {
                        "language": self.language,
                        "duration": 1.0,
                        "timestamp": time.time(),
                    },
                )
            time.sleep(5)

    def _file_audio_loop(self):
        logger.info("🎧 Loading audio from fallback test file")
        test_file = os.getenv("ALPHA_VOX_TEST_AUDIO", "test_input.wav")
        if not os.path.exists(test_file):
            logger.warning(f"❌ Test file not found: {test_file}")
            return
        with sr.AudioFile(test_file) as source:
            audio = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            logger.info(f"📁 Recognized from file: {text}")
            for cb in self.callbacks:
                cb(text, 0.99, {"mode": "file", "timestamp": time.time()})
        except Exception as e:
            logger.error(f"❌ File recognition error: {e}")
            for cb in self.callbacks:
                cb("", 0.0, {"error": str(e)})

    def _microphone_loop(self):
        try:
            mic = sr.Microphone(device_index=self.device_index)
            with mic as source:
                self.recognizer.energy_threshold = 300  # Static threshold
                logger.info(
                    f"🎧 Listening started (PyAudio Mic) with threshold {self.recognizer.energy_threshold}"
                )
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=None)
                        with open("debug_input.wav", "wb") as f:
                            f.write(audio.get_wav_data())

                        text = self.recognizer.recognize_google(
                            audio, language=self.language
                        )
                        logger.info(f"[Recognized] {text}")
                        for cb in self.callbacks:
                            cb(text, 0.9, {"mode": "live", "timestamp": time.time()})
                    except sr.UnknownValueError:
                        logger.warning("⚠️ Could not understand audio")
                        for cb in self.callbacks:
                            cb("", 0.0, {"error": "unrecognized"})
                    except Exception as e:
                        logger.error(f"❌ Mic recognition error: {e}")
                        for cb in self.callbacks:
                            cb("", 0.0, {"error": str(e)})
        except Exception as e:
            logger.error(f"❌ Microphone loop error: {e}")
            for cb in self.callbacks:
                cb("", 0.0, {"error": str(e)})


# Singleton accessor
_speech_recognition_engine = None


def get_speech_recognition_engine(
    simulate: bool = False, device_index: int = None
) -> SpeechRecognitionEngine:
    global _speech_recognition_engine
    if _speech_recognition_engine is None:
        _speech_recognition_engine = SpeechRecognitionEngine(
            simulate=simulate, device_index=device_index
        )
    return _speech_recognition_engine


def list_microphones():
    return sr.Microphone.list_microphone_names()


if __name__ == "__main__":
    print("Available microphones:")
    for i, name in enumerate(list_microphones()):
        print(f"{i}: {name}")

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
