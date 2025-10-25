"""Action scheduling utilities for Derek autonomy."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4


logger = logging.getLogger(__name__)


class ActionScheduler:
    """Persists follow-up actions derived from reflections and requests."""

    def __init__(
        self, memory_engine, action_file: str = "logs/autonomy_actions.json"
    ) -> None:
        self.memory_engine = memory_engine
        self.path = Path(action_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_actions(self) -> List[Dict[str, any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Failed to decode %s; starting with empty queue", self.path)
            return []

    def _save_actions(self, actions: List[Dict[str, any]]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2)

    def create_action(
        self,
        description: str,
        *,
        source: str = "system",
        requires_approval: bool = False,
        tags: Optional[List[str]] = None,
    ) -> str:
        actions = self._load_actions()
        for action in actions:
            if action["description"] == description and action["status"] == "pending":
                return action["id"]
        action_id = str(uuid4())
        action = {
            "id": action_id,
            "description": description,
            "source": source,
            "requires_approval": requires_approval,
            "status": "pending",
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
        }
        actions.append(action)
        self._save_actions(actions)
        return action_id

    def complete_action(
        self,
        action_id: str,
        *,
        status: str = "completed",
        notes: Optional[str] = None,
    ) -> bool:
        actions = self._load_actions()
        updated = False
        for action in actions:
            if action["id"] == action_id:
                action["status"] = status
                action["completed_at"] = datetime.utcnow().isoformat()
                if notes:
                    action.setdefault("notes", []).append(notes)
                updated = True
                break
        if updated:
            self._save_actions(actions)
        else:
            logger.warning("Action %s not found when updating status", action_id)
        return updated

    def sync_suggestions(
        self, suggestions: List[str], *, source: str = "planner"
    ) -> List[str]:
        action_ids = []
        for suggestion in suggestions:
            action_ids.append(
                self.create_action(
                    suggestion,
                    source=source,
                    requires_approval=True,
                    tags=["reflection"],
                )
            )
        return action_ids

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
