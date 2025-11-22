"""
Configuration for Avatar Video App
"""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    # Look for .env in parent directories
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try default locations
except ImportError:
    pass  # python-dotenv not installed

@dataclass
class AvatarConfig:
    """Configuration settings for avatar generation"""
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    did_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None

    # Avatar settings
    default_avatar_size: tuple = (512, 512)
    avatar_styles: list = None

    # Video settings
    max_video_duration: int = 180  # 3 minutes in seconds
    video_fps: int = 30
    video_resolution: tuple = (1080, 1920)  # Portrait mode
    output_format: str = "mp4"

    # Voice settings
    sample_rate: int = 44100
    audio_channels: int = 1
    audio_format: str = "wav"

    # Voice changer presets
    voice_effects: list = None

    # Storage paths
    temp_dir: str = "/tmp/avatar_video_app"
    output_dir: str = "./avatar_outputs"

    def __post_init__(self):
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.did_api_key = os.getenv("DID_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Default avatar styles
        if self.avatar_styles is None:
            self.avatar_styles = [
                "realistic",
                "anime",
                "cartoon",
                "3d_render",
                "artistic",
                "pixel_art",
                "watercolor",
                "oil_painting"
            ]

        # Default voice effects
        if self.voice_effects is None:
            self.voice_effects = [
                {"name": "normal", "pitch": 1.0, "speed": 1.0, "reverb": 0.0},
                {"name": "deep", "pitch": 0.7, "speed": 0.95, "reverb": 0.1},
                {"name": "high", "pitch": 1.4, "speed": 1.05, "reverb": 0.0},
                {"name": "robot", "pitch": 1.0, "speed": 1.0, "reverb": 0.3, "vocoder": True},
                {"name": "echo", "pitch": 1.0, "speed": 1.0, "reverb": 0.6},
                {"name": "whisper", "pitch": 1.1, "speed": 0.9, "reverb": 0.2, "breathiness": 0.7},
                {"name": "chipmunk", "pitch": 1.8, "speed": 1.2, "reverb": 0.0},
                {"name": "villain", "pitch": 0.6, "speed": 0.85, "reverb": 0.4},
                {"name": "announcer", "pitch": 0.85, "speed": 0.9, "reverb": 0.2},
                {"name": "ethereal", "pitch": 1.2, "speed": 0.95, "reverb": 0.5, "chorus": True}
            ]

        # Ensure directories exist
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# Global config instance
config = AvatarConfig()
