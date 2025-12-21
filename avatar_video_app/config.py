"""
Configuration for Avatar Video App
"""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Load environment variables from .env file
env_loaded = False
try:
    from dotenv import load_dotenv
    # Try multiple .env file locations
    possible_paths = [
        Path(__file__).parent.parent / '.env',  # Project root
        Path(__file__).parent / '.env',          # avatar_video_app folder
        Path.cwd() / '.env',                     # Current working directory
    ]

    for env_path in possible_paths:
        if env_path.exists():
            print(f"[CONFIG] Loading .env from: {env_path}")
            load_dotenv(env_path, override=True)
            env_loaded = True
            print(f"[CONFIG] .env loaded successfully from {env_path}")
            break

    if not env_loaded:
        print(f"[CONFIG] No .env file found in searched locations")

except ImportError:
    print(f"[CONFIG] python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"[CONFIG] Error loading .env: {e}")

@dataclass
class AvatarConfig:
    """Configuration settings for avatar generation"""
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    did_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    hedra_api_key: Optional[str] = None
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

    # Storage paths
    temp_dir: str = "/tmp/avatar_video_app"
    output_dir: str = "./avatar_outputs"

    def __post_init__(self):
        # Load API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.did_api_key = os.getenv("DID_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.hedra_api_key = os.getenv("HEDRA_API_KEY")
        self.heygen_api_key = os.getenv("HEYGEN_API_KEY")

        # Debug: Show ALL environment variables with "HEDRA" in them
        print(f"[CONFIG] All HEDRA environment variables:")
        for key, value in os.environ.items():
            if "HEDRA" in key.upper():
                print(f"[CONFIG]   {key} = {value[:20]}..." if len(value) > 20 else f"[CONFIG]   {key} = {value}")

        # Debug: Show which keys are loaded
        print(f"[CONFIG] HEDRA_API_KEY loaded: {'Yes' if self.hedra_api_key else 'No'}")
        if self.hedra_api_key:
            print(f"[CONFIG] HEDRA_API_KEY: {self.hedra_api_key[:10]}...")

        print(f"[CONFIG] HEYGEN_API_KEY loaded: {'Yes' if self.heygen_api_key else 'No'}")
        if self.heygen_api_key:
            print(f"[CONFIG] HEYGEN_API_KEY: {self.heygen_api_key[:10]}...")

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
