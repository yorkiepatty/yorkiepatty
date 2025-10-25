# derek_identity.py
import os
import json
import logging
from conversation_engine import ConversationEngine
from memory_engine import MemoryEngine
import yaml
from pathlib import Path

# Resolve path relative to this file’s directory
manifest_path = Path(__file__).resolve().parent / "derek_manifest.yaml"

if not manifest_path.exists():
    raise FileNotFoundError(f"❌ Derek manifest not found at {manifest_path}")

with open(manifest_path, "r", encoding="utf-8") as file:
    derek_manifest = yaml.safe_load(file)

logger = logging.getLogger(__name__)


class Derek:
    def __init__(self, identity_path="derek_identity.json"):
        self.identity = self.load_identity(identity_path)

        # pull memory path from manifest
        memory_path = derek_manifest.get("memory_path", "./memory/memory_store.json")
        os.makedirs(memory_path, exist_ok=True)

        # initialize memory engine with path
        self.memory_engine = MemoryEngine(
            file_path=os.path.join(memory_path, "context.json")
        )
        self.conversation_engine = ConversationEngine()

        logger.info(f"✅ Derek initialized with memory engine at: {memory_path}")

        "Derek initialized with identity: %s", self.identity.get("name", "Unknown")

    def load_identity(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def think(self, input_text):
        return {"output": f"I heard: {input_text}"}


# ✅ Global instance for import
derek = Derek()

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
