"""
Royce Configuration
-------------------
Central configuration for all Royce systems.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()


class RoyceConfig:
    """All configuration for Royce AI Assistant"""

    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MEMORY_DIR = PROJECT_ROOT / "data" / "memory"
    LOGS_DIR = PROJECT_ROOT / "data" / "logs"
    IMAGES_DIR = PROJECT_ROOT / "data" / "images"
    MUSIC_CACHE_DIR = PROJECT_ROOT / "data" / "music_cache"

    # Identity
    IDENTITY_FILE = PROJECT_ROOT / "royce_identity.json"

    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    # AI Model
    AI_PROVIDER = os.getenv("ROYCE_AI_PROVIDER", "anthropic")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")

    # Voice — ElevenLabs (primary), AWS Polly (fallback)
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
    ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs")
    VOICE_ID = os.getenv("ROYCE_VOICE", "Matthew")
    POLLY_ENGINE = os.getenv("POLLY_ENGINE", "neural")

    # Speech Recognition
    SPEECH_BACKEND = os.getenv("SPEECH_BACKEND", "google")
    MIC_ENERGY_THRESHOLD = int(os.getenv("MIC_ENERGY_THRESHOLD", "3000"))
    LISTEN_TIMEOUT = int(os.getenv("LISTEN_TIMEOUT", "20"))
    PHRASE_TIME_LIMIT = int(os.getenv("PHRASE_TIME_LIMIT", "60"))
    PAUSE_THRESHOLD = float(os.getenv("PAUSE_THRESHOLD", "2.0"))

    # Memory
    MEMORY_DB = str(MEMORY_DIR / "royce_memory.db")
    MAX_MEMORY_ENTRIES = int(os.getenv("MAX_MEMORY_ENTRIES", "50000"))
    MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "36500"))  # ~100 years = forever

    # News
    NEWS_REFRESH_INTERVAL = int(os.getenv("NEWS_REFRESH_MINUTES", "15"))
    NEWS_SOURCES = os.getenv("NEWS_SOURCES", "bbc-news,cnn,reuters,associated-press,the-verge,techcrunch")

    # Conversation
    MAX_CONVERSATION_HISTORY = 200
    RESPONSE_PATIENCE_SECONDS = float(os.getenv("RESPONSE_PATIENCE", "2.5"))

    @classmethod
    def ensure_directories(cls):
        """Create all required directories"""
        for d in [cls.DATA_DIR, cls.MEMORY_DIR, cls.LOGS_DIR, cls.IMAGES_DIR, cls.MUSIC_CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load_identity(cls) -> Dict[str, Any]:
        """Load Royce's identity file"""
        with open(cls.IDENTITY_FILE, "r") as f:
            return json.load(f)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        return getattr(cls, key, default)


# Create directories on import
RoyceConfig.ensure_directories()
