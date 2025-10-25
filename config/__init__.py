"""
Derek Configuration Package
The Christman AI Project

Configuration management for Derek's consciousness system
"""

from .settings import (
    get_setting,
    update_setting,
    get_all_settings,
    DEREK_SETTINGS,
    AUDIO_SETTINGS,
    LEARNING_SETTINGS,
    API_SETTINGS,
    MEMORY_SETTINGS,
    SERVICE_SETTINGS,
    HARDWARE_SETTINGS,
    LOGGING_SETTINGS,
    SECURITY_SETTINGS,
    API_KEYS
)

__all__ = [
    "get_setting",
    "update_setting", 
    "get_all_settings",
    "DEREK_SETTINGS",
    "AUDIO_SETTINGS",
    "LEARNING_SETTINGS",
    "API_SETTINGS",
    "MEMORY_SETTINGS",
    "SERVICE_SETTINGS",
    "HARDWARE_SETTINGS",
    "LOGGING_SETTINGS",
    "SECURITY_SETTINGS",
    "API_KEYS"
]