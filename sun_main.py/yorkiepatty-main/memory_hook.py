import json
import time
from pathlib import Path
from typing import Dict, Any, List

MEMORY_FILE = Path("memory/session_memory.json")
MAX_CONTEXT = 10  # how many exchanges to keep

def _load() -> List[Dict[str, Any]]:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            return []
    return []

def _save(data: List[Dict[str, Any]]):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(data, indent=2))

def remember(role: str, content: str):
    """Append one line of dialogue or event."""
    data = _load()
    data.append({"time": time.time(), "role": role, "content": content})
    _save(data[-MAX_CONTEXT:])  # keep it short

def recall() -> str:
    """Return formatted short-term memory context."""
    data = _load()
    lines = [f"{d['role'].upper()}: {d['content']}" for d in data[-MAX_CONTEXT:]]
    return "\n".join(lines)

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
