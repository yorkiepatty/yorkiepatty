"""
Core Module
-----------
Core utilities and shared functionality for Derek's modules.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("core")


# Memory hook functions
def remember(key: str, value: Any, category: str = "general"):
    """Store information in Derek's memory"""
    try:
        from memory_engine import MemoryEngine
        memory = MemoryEngine()
        memory.add_memory(key, value, category)
        logger.debug(f"Remembered: {key} in {category}")
    except Exception as e:
        logger.error(f"Failed to remember: {e}")


def recall(key: str, default: Any = None) -> Any:
    """Retrieve information from Derek's memory"""
    try:
        from memory_engine import MemoryEngine
        memory = MemoryEngine()
        result = memory.get_memory(key)
        return result if result is not None else default
    except Exception as e:
        logger.error(f"Failed to recall: {e}")
        return default


# Path utilities
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = PROJECT_ROOT / "memory"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MEMORY_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Configuration helper
class Config:
    """Configuration manager"""
    
    def __init__(self):
        self.values = {}
        self.load_from_env()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        self.values = {
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'PERPLEXITY_API_KEY': os.getenv('PERPLEXITY_API_KEY'),
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'PROJECT_ROOT': str(PROJECT_ROOT),
            'DATA_DIR': str(DATA_DIR),
            'MEMORY_DIR': str(MEMORY_DIR),
            'LOGS_DIR': str(LOGS_DIR),
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.values.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.values[key] = value


# Global config instance
config = Config()


# Module initialization helper
def ensure_path_in_sys(path: str):
    """Ensure a path is in sys.path"""
    if path not in sys.path:
        sys.path.insert(0, path)
        logger.debug(f"Added to sys.path: {path}")


# Initialize project root in path
ensure_path_in_sys(str(PROJECT_ROOT))


__all__ = [
    'remember',
    'recall',
    'Config',
    'config',
    'PROJECT_ROOT',
    'DATA_DIR',
    'MEMORY_DIR',
    'LOGS_DIR',
    'ensure_path_in_sys'
]

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
