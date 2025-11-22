"""
Voice Processor - Handles voice recording, processing, and effects
Includes voice changer functionality with multiple effects
"""
import os
import io
import base64
import wave
import struct
import math
import json
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import tempfile
import hashlib
from datetime import datetime

from .config import config


@dataclass
class VoiceResult:
    """Result from voice processing"""
    success: bool
    audio_path: Optional[str] = None
    audio_base64: Optional[str] = None
    duration: float = 0.0
    sample_rate: int = 44100
    effect_applied: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VoiceProcessor:
    """
    Processes voice audio with various effects and transformations.
    Supports recording, playback, and voice changing.
    """

    def __init__(self):
        self.config = config
        self.temp_dir = Path(config.temp_dir) / "voice"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.effects = {e["name"]: e for e in config.voice_effects}

        # Try to import audio libraries
        self._init_audio_libs()

    def _init_audio_libs(self):
        """Initialize audio processing libraries"""
        self.numpy_available = False
        self.scipy_available = False
        self.librosa_available = False

        try:
            import numpy as np
            self.np = np
            self.numpy_available = True
        except ImportError:
            pass

        try:
            from scipy import signal
            from scipy.io import wavfile
            self.signal = signal
            self.wavfile = wavfile
            self.scipy_available = True
        except ImportError:
            pass

        try:
            import librosa
            import soundfile as sf
            self.librosa = librosa
            self.soundfile = sf
            self.librosa_available = True
        except ImportError:
            pass

    def get_available_effects(self) -> List[Dict[str, Any]]:
        """Get list of available voice effects"""
        return config.voice_effects

    def _generate_audio_id(self) -> str:
        """Generate unique ID for audio file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.md5(os.urandom(8)).hexdigest()[:8]
        return f"voice_{timestamp}_{random_part}"

    async def process_uploaded_audio(
        self,
        audio_data: bytes,
        effect_name: str = "normal",
        input_format: str = "wav"
    ) -> VoiceResult:
        """
        Process uploaded audio data with optional effects.

        Args:
            audio_data: Raw audio bytes
            effect_name: Name of voice effect to apply
            input_format: Input audio format (wav, mp3, webm, etc.)

        Returns:
            VoiceResult with processed audio
        """
        try:
            audio_id = self._generate_audio_id()

            # Save input audio
            input_path = self.temp_dir / f"{audio_id}_input.{input_format}"
            with open(input_path, "wb") as f:
                f.write(audio_data)

            # Convert to WAV if needed
            if input_format != "wav":
                wav_path = await self._convert_to_wav(input_path)
                if not wav_path:
                    return VoiceResult(
                        success=False,
                        error=f"Failed to convert {input_format} to WAV"
                    )
            else:
                wav_path = input_path

            # Apply voice effect
            if effect_name and effect_name != "normal":
                processed_path = await self._apply_effect(wav_path, effect_name, audio_id)
            else:
                processed_path = wav_path

            # Get audio info
            duration = self._get_audio_duration(processed_path)

            # Read processed audio
            with open(processed_path, "rb") as f:
                processed_data = f.read()

            audio_b64 = base64.b64encode(processed_data).decode()

            return VoiceResult(
                success=True,
                audio_path=str(processed_path),
                audio_base64=audio_b64,
                duration=duration,
                sample_rate=config.sample_rate,
                effect_applied=effect_name,
                metadata={
                    "audio_id": audio_id,
                    "input_format": input_format,
                    "effect": effect_name
                }
            )

        except Exception as e:
            return VoiceResult(success=False, error=f"Audio processing error: {str(e)}")

    async def _convert_to_wav(self, input_path: Path) -> Optional[Path]:
        """Convert audio file to WAV format"""
        output_path = input_path.with_suffix(".wav")

        # Try using ffmpeg
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", str(input_path), "-ar", str(config.sample_rate),
                "-ac", str(config.audio_channels), "-y", str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()

            if output_path.exists():
                return output_path
        except Exception:
            pass

        # Try using librosa
        if self.librosa_available:
            try:
                y, sr = self.librosa.load(str(input_path), sr=config.sample_rate, mono=True)
                self.soundfile.write(str(output_path), y, config.sample_rate)
                return output_path
            except Exception:
                pass

        return None

    async def _apply_effect(
        self,
        audio_path: Path,
        effect_name: str,
        audio_id: str
    ) -> Path:
        """Apply voice effect to audio file"""
        output_path = self.temp_dir / f"{audio_id}_processed.wav"

        effect = self.effects.get(effect_name, self.effects["normal"])

        if self.librosa_available and self.numpy_available:
            return await self._apply_effect_librosa(audio_path, effect, output_path)
        elif self.scipy_available and self.numpy_available:
            return await self._apply_effect_scipy(audio_path, effect, output_path)
        else:
            return await self._apply_effect_basic(audio_path, effect, output_path)

    async def _apply_effect_librosa(
        self,
        audio_path: Path,
        effect: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Apply effect using librosa"""
        try:
            # Load audio
            y, sr = self.librosa.load(str(audio_path), sr=config.sample_rate, mono=True)

            # Apply pitch shift
            pitch_factor = effect.get("pitch", 1.0)
            if pitch_factor != 1.0:
                n_steps = 12 * self.np.log2(pitch_factor)
                y = self.librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

            # Apply time stretch (speed change)
            speed_factor = effect.get("speed", 1.0)
            if speed_factor != 1.0:
                y = self.librosa.effects.time_stretch(y, rate=speed_factor)

            # Apply reverb (simple convolution)
            reverb_amount = effect.get("reverb", 0.0)
            if reverb_amount > 0:
                impulse_length = int(sr * 0.5 * reverb_amount)
                impulse = self.np.exp(-self.np.linspace(0, 5, impulse_length))
                impulse = impulse / impulse.sum()
                y_reverb = self.np.convolve(y, impulse, mode='full')[:len(y)]
                y = (1 - reverb_amount * 0.5) * y + reverb_amount * 0.5 * y_reverb

            # Apply breathiness (add noise)
            breathiness = effect.get("breathiness", 0.0)
            if breathiness > 0:
                noise = self.np.random.randn(len(y)) * breathiness * 0.1
                y = y + noise

            # Apply vocoder effect (robot voice)
            if effect.get("vocoder"):
                # Simple vocoder simulation
                window = 512
                hop = 128
                stft = self.librosa.stft(y, n_fft=window, hop_length=hop)
                magnitude = self.np.abs(stft)
                phase = self.np.exp(1j * self.np.random.uniform(0, 2*self.np.pi, stft.shape))
                y = self.librosa.istft(magnitude * phase, hop_length=hop)

            # Apply chorus effect
            if effect.get("chorus"):
                delay_samples = int(sr * 0.025)
                depth = 0.002 * sr
                rate = 1.5
                t = self.np.arange(len(y)) / sr
                modulation = depth * self.np.sin(2 * self.np.pi * rate * t)
                y_chorus = self.np.zeros_like(y)
                for i in range(len(y)):
                    delay = int(delay_samples + modulation[i])
                    if i - delay >= 0:
                        y_chorus[i] = 0.5 * y[i] + 0.5 * y[i - delay]
                    else:
                        y_chorus[i] = y[i]
                y = y_chorus

            # Normalize
            y = y / (self.np.max(self.np.abs(y)) + 1e-8) * 0.95

            # Save
            self.soundfile.write(str(output_path), y, config.sample_rate)
            return output_path

        except Exception as e:
            print(f"Librosa effect error: {e}")
            # Fall back to copying original
            import shutil
            shutil.copy(str(audio_path), str(output_path))
            return output_path

    async def _apply_effect_scipy(
        self,
        audio_path: Path,
        effect: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Apply effect using scipy"""
        try:
            sr, data = self.wavfile.read(str(audio_path))

            # Convert to float
            if data.dtype == self.np.int16:
                data = data.astype(self.np.float32) / 32768.0
            elif data.dtype == self.np.int32:
                data = data.astype(self.np.float32) / 2147483648.0

            # Make mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # Apply pitch shift (resample method)
            pitch_factor = effect.get("pitch", 1.0)
            speed_factor = effect.get("speed", 1.0)

            combined_factor = pitch_factor * speed_factor
            if combined_factor != 1.0:
                new_length = int(len(data) / combined_factor)
                data = self.signal.resample(data, new_length)

            # Apply reverb
            reverb_amount = effect.get("reverb", 0.0)
            if reverb_amount > 0:
                impulse_length = int(sr * 0.3 * reverb_amount)
                impulse = self.np.exp(-self.np.linspace(0, 4, impulse_length))
                impulse = impulse / impulse.sum()
                data_reverb = self.signal.convolve(data, impulse, mode='full')[:len(data)]
                data = (1 - reverb_amount * 0.5) * data + reverb_amount * 0.5 * data_reverb

            # Normalize
            data = data / (self.np.max(self.np.abs(data)) + 1e-8) * 0.95

            # Convert back to int16
            data_int = (data * 32767).astype(self.np.int16)

            # Save
            self.wavfile.write(str(output_path), config.sample_rate, data_int)
            return output_path

        except Exception as e:
            print(f"Scipy effect error: {e}")
            import shutil
            shutil.copy(str(audio_path), str(output_path))
            return output_path

    async def _apply_effect_basic(
        self,
        audio_path: Path,
        effect: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Apply basic effect using pure Python (limited functionality)"""
        try:
            with wave.open(str(audio_path), 'rb') as wav_in:
                params = wav_in.getparams()
                frames = wav_in.readframes(params.nframes)

            # Convert to samples
            samples = list(struct.unpack(f'{params.nframes}h', frames))

            # Apply basic pitch shift via resampling
            pitch_factor = effect.get("pitch", 1.0)
            if pitch_factor != 1.0:
                new_length = int(len(samples) / pitch_factor)
                new_samples = []
                for i in range(new_length):
                    idx = min(int(i * pitch_factor), len(samples) - 1)
                    new_samples.append(samples[idx])
                samples = new_samples

            # Normalize
            max_val = max(abs(s) for s in samples) or 1
            samples = [int(s * 32000 / max_val) for s in samples]

            # Write output
            with wave.open(str(output_path), 'wb') as wav_out:
                wav_out.setparams(params)
                wav_out.setnframes(len(samples))
                frames_out = struct.pack(f'{len(samples)}h', *samples)
                wav_out.writeframes(frames_out)

            return output_path

        except Exception as e:
            print(f"Basic effect error: {e}")
            import shutil
            shutil.copy(str(audio_path), str(output_path))
            return output_path

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds"""
        try:
            with wave.open(str(audio_path), 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0

    async def text_to_speech(
        self,
        text: str,
        voice: str = "default",
        effect_name: str = "normal"
    ) -> VoiceResult:
        """
        Convert text to speech audio.

        Args:
            text: Text to convert to speech
            voice: Voice to use (implementation-dependent)
            effect_name: Voice effect to apply

        Returns:
            VoiceResult with generated audio
        """
        audio_id = self._generate_audio_id()

        try:
            # Try gTTS first
            from gtts import gTTS

            output_path = self.temp_dir / f"{audio_id}_tts.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(str(output_path))

            # Convert to WAV
            wav_path = await self._convert_to_wav(output_path)
            if not wav_path:
                wav_path = output_path

            # Apply effect if needed
            if effect_name and effect_name != "normal":
                final_path = await self._apply_effect(wav_path, effect_name, audio_id)
            else:
                final_path = wav_path

            duration = self._get_audio_duration(final_path)

            with open(final_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()

            return VoiceResult(
                success=True,
                audio_path=str(final_path),
                audio_base64=audio_b64,
                duration=duration,
                sample_rate=config.sample_rate,
                effect_applied=effect_name,
                metadata={
                    "audio_id": audio_id,
                    "source": "tts",
                    "voice": voice,
                    "text_length": len(text)
                }
            )

        except ImportError:
            return VoiceResult(
                success=False,
                error="Text-to-speech not available. Install gTTS: pip install gTTS"
            )
        except Exception as e:
            return VoiceResult(success=False, error=f"TTS error: {str(e)}")

    def validate_audio_duration(self, duration: float) -> Tuple[bool, str]:
        """Validate audio duration for video generation"""
        max_duration = config.max_video_duration

        if duration <= 0:
            return False, "Audio duration must be greater than 0"
        if duration > max_duration:
            return False, f"Audio duration ({duration:.1f}s) exceeds maximum ({max_duration}s)"

        return True, f"Audio duration: {duration:.1f}s (max: {max_duration}s)"


# Singleton instance
voice_processor = VoiceProcessor()
