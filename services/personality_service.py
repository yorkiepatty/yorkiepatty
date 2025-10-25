"""Personality management for Derek Dashboard."""

import logging
from typing import Dict, Any

from config.settings import Settings

logger = logging.getLogger(__name__)


class PersonalityService:
    """Provides access to Derek's personality profile."""

    def __init__(self):
        self.settings = Settings()
        self.profile: Dict[str, Any] = {}

    def load_profile(self) -> Dict[str, Any]:
        """Load the personality profile from settings."""
        self.profile = self.settings.identity
        logger.info(
            "Personality profile loaded for %s", self.profile.get("name", "Derek")
        )
        return self.profile

        # In start():
        self.personality.load_profile()  # ← LOADS Derek's character config

    def get_trait(self, trait_name: str) -> Any:
        """Retrieve a specific trait from the profile."""
        if not self.profile:
            self.load_profile()

        traits = self.profile.get("personality", {}).get("core_traits", [])
        return trait_name in traits

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
