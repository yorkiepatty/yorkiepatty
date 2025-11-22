"""
Configuration for Avatar Video App
"""
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Load .env file with verbose output
print("\n" + "="*60)
print("[CONFIG] Loading configuration...")

env_loaded = False
env_file_used = None

try:
    from dotenv import load_dotenv

    # Try multiple locations for .env file
    possible_paths = [
        Path(__file__).parent.parent / '.env',  # yorkiepatty/.env
        Path.cwd() / '.env',  # Current working directory
        Path.home() / '.env',  # User home directory
    ]

    for env_path in possible_paths:
        print(f"[CONFIG] Checking for .env at: {env_path}")
        if env_path.exists():
            print(f"[CONFIG] FOUND .env file at: {env_path}")
            load_dotenv(env_path, override=True)
            env_loaded = True
            env_file_used = env_path
            break

    if not env_loaded:
        print("[CONFIG] No .env file found, using environment variables only")
        load_dotenv()  # Try default locations

except ImportError:
    print("[CONFIG] WARNING: python-dotenv not installed, using environment variables only")

# Show what was loaded
api_key = os.getenv("OPENAI_API_KEY", "")
if api_key and api_key.startswith("sk-"):
    print(f"[CONFIG] OpenAI API Key: LOADED ({api_key[:12]}...)")
elif api_key:
    print(f"[CONFIG] OpenAI API Key: INVALID FORMAT (starts with: {api_key[:10]}...)")
else:
    print("[CONFIG] OpenAI API Key: NOT SET")
print("="*60 + "\n")

@dataclass
class AvatarConfig:
    """Configuration settings for avatar generation"""
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    did_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    heygen_api_key: Optional[str] = None

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

    # Storage paths (auto-detect based on OS)
    temp_dir: str = None
    output_dir: str = "./avatar_outputs"

    def __post_init__(self):
        # Set temp directory based on OS
        if self.temp_dir is None:
            self.temp_dir = os.path.join(tempfile.gettempdir(), "avatar_video_app")

        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.did_api_key = os.getenv("DID_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.heygen_api_key = os.getenv("HEYGEN_API_KEY")

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
