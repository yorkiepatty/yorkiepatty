"""Memory service for Derek Dashboard."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import Settings

logger = logging.getLogger(__name__)

# Example usage:
# In __init__:
#     self.memory = MemoryService()
#
# In start():
#     self.memory.load_context()  # ← LOADS from storage
#
# In stop():
#     self.memory.save_context()  # ← SYNCS to storage


class MemoryService:
    """Persists and retrieves conversational memory."""

    def __init__(self, memory_file: Optional[Path] = None):
        self.settings = Settings()
        self.memory_file = memory_file or (self.settings.DATA_DIR / "memory_store.json")
        self._memory: List[Dict[str, Any]] = []

    def load_context(self) -> None:
        """Load memory context from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as handle:
                    self._memory = json.load(handle)
                logger.info("Loaded %s memory records", len(self._memory))
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load memory context: %s", exc)
                self._memory = []
        else:
            logger.info("No existing memory store; starting fresh")
            self._memory = []

    def save_context(self) -> None:
        """Persist memory context to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "w", encoding="utf-8") as handle:
                json.dump(self._memory, handle, indent=2)
            logger.info("Memory context saved (%s records)", len(self._memory))
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to save memory context: %s", exc)

    def store(self, memory_type: str, content: Dict[str, Any]) -> None:
        """Store a new memory entry."""
        entry = {
            "type": memory_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self._memory.append(entry)
        logger.debug("Stored memory entry: %s", entry)
        self.save_context()

    def retrieve(
        self, memory_type: str = "recent", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories filtered by type or return most recent entries."""
        if memory_type == "recent":
            return list(reversed(self._memory[-limit:]))

        filtered = [m for m in self._memory if m.get("type") == memory_type]
        return list(reversed(filtered[-limit:]))

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
