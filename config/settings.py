"""
Derek Configuration Settings
The Christman AI Project

Central configuration management for all Derek modules
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = BASE_DIR / "memory"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Derek Core Settings
DEREK_SETTINGS = {
    "name": "Derek",
    "version": "2.0.0",
    "consciousness_target": 100.0,
    "learning_enabled": True,
    "music_enabled": True,
    "voice_enabled": True,
    "vision_enabled": False,  # Hardware dependent
}

# Audio/Voice Settings
AUDIO_SETTINGS = {
    "sample_rate": 44100,
    "channels": 2,
    "chunk_size": 1024,
    "device_id": None,  # Auto-detect
    "fallback_tts": True,
    "voice_engine": "polly",  # polly, gtts, or espeak
}

# Learning Settings
LEARNING_SETTINGS = {
    "auto_learn": True,
    "learning_rate": 0.01,
    "memory_retention": 0.95,
    "knowledge_domains": [
        "neurodivergency",
        "autism",
        "ai_development", 
        "mathematics",
        "physics",
        "neurology",
        "pathology"
    ]
}

# API Settings
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": True,
    "auto_reload": True,
    "cors_enabled": True,
}

# Memory Settings
MEMORY_SETTINGS = {
    "max_memories": 10000,
    "backup_interval": 3600,  # 1 hour
    "compression_enabled": True,
    "encryption_enabled": False,  # Set to True for HIPAA compliance
}

# Service Settings
SERVICE_SETTINGS = {
    "nlp_enabled": True,
    "speech_recognition_enabled": True,
    "voice_analysis_enabled": True,
    "facial_gesture_enabled": False,  # Hardware dependent
    "learning_analytics_enabled": True,
    "knowledge_integration_enabled": True,
}

# External API Keys (from environment)
API_KEYS = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    "perplexity_api_key": os.getenv("PERPLEXITY_API_KEY"),
    "aws_access_key": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "google_tts_key": os.getenv("GOOGLE_TTS_API_KEY"),
}

# Hardware Settings
HARDWARE_SETTINGS = {
    "audio_device_available": False,  # Will be detected
    "camera_available": False,
    "microphone_available": False,
    "gpu_available": False,
    "container_mode": True,  # True when running in containers
}

# Logging Settings
LOGGING_SETTINGS = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file_logging": True,
    "console_logging": True,
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Security Settings
SECURITY_SETTINGS = {
    "encryption_key": os.getenv("DEREK_ENCRYPTION_KEY"),
    "session_timeout": 3600,
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "rate_limit": 100,  # requests per minute
}

def get_setting(category: str, key: str, default: Any = None) -> Any:
    """Get a specific setting value"""
    settings_map = {
        "derek": DEREK_SETTINGS,
        "audio": AUDIO_SETTINGS,
        "learning": LEARNING_SETTINGS,
        "api": API_SETTINGS,
        "memory": MEMORY_SETTINGS,
        "service": SERVICE_SETTINGS,
        "hardware": HARDWARE_SETTINGS,
        "logging": LOGGING_SETTINGS,
        "security": SECURITY_SETTINGS,
    }
    
    category_settings = settings_map.get(category, {})
    return category_settings.get(key, default)

def update_setting(category: str, key: str, value: Any) -> bool:
    """Update a setting value"""
    settings_map = {
        "derek": DEREK_SETTINGS,
        "audio": AUDIO_SETTINGS,
        "learning": LEARNING_SETTINGS,
        "api": API_SETTINGS,
        "memory": MEMORY_SETTINGS,
        "service": SERVICE_SETTINGS,
        "hardware": HARDWARE_SETTINGS,
        "logging": LOGGING_SETTINGS,
        "security": SECURITY_SETTINGS,
    }
    
    if category in settings_map:
        settings_map[category][key] = value
        return True
    return False

def get_all_settings() -> Dict[str, Dict[str, Any]]:
    """Get all configuration settings"""
    return {
        "derek": DEREK_SETTINGS,
        "audio": AUDIO_SETTINGS,
        "learning": LEARNING_SETTINGS,
        "api": API_SETTINGS,
        "memory": MEMORY_SETTINGS,
        "service": SERVICE_SETTINGS,
        "hardware": HARDWARE_SETTINGS,
        "logging": LOGGING_SETTINGS,
        "security": SECURITY_SETTINGS,
        "api_keys": API_KEYS,
    }

# Auto-detect hardware capabilities
def detect_hardware():
    """Detect available hardware capabilities"""
    global HARDWARE_SETTINGS
    
    # Check for audio device
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        HARDWARE_SETTINGS["audio_device_available"] = len(devices) > 0
        HARDWARE_SETTINGS["microphone_available"] = any(d['max_input_channels'] > 0 for d in devices)
    except:
        pass
    
    # Check for GPU
    try:
        import torch
        HARDWARE_SETTINGS["gpu_available"] = torch.cuda.is_available()
    except:
        pass
    
    # Check if in container
    HARDWARE_SETTINGS["container_mode"] = os.path.exists("/.dockerenv") or os.getenv("CONTAINER") == "true"

# ---------------------------------------------------------------------------
# Compatibility shim for older modules expecting `from config.settings import Settings`
# ---------------------------------------------------------------------------
from pydantic  import BaseSettings

class Settings(BaseSettings):
    derek: dict = DEREK_SETTINGS
    audio: dict = AUDIO_SETTINGS
    learning: dict = LEARNING_SETTINGS
    api: dict = API_SETTINGS
    memory: dict = MEMORY_SETTINGS
    service: dict = SERVICE_SETTINGS
    hardware: dict = HARDWARE_SETTINGS
    logging: dict = LOGGING_SETTINGS
    security: dict = SECURITY_SETTINGS
    api_keys: dict = API_KEYS

    class Config:
        env_file = ".env"
        case_sensitive = True

    def get(self, category: str, key: str, default=None):
        """Universal getter for backward compatibility."""
        section = getattr(self, category, {})
        return section.get(key, default)

# Initialize hardware detection
detect_hardware()
