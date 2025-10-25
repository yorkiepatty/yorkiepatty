"""UI module registration for Derek Dashboard."""

from typing import Callable, Dict


class UIModuleRegistry:
    """Stores UI component factories."""

    def __init__(self):
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, factory: Callable) -> None:
        self._registry[name] = factory

    def get(self, name: str) -> Callable:
        return self._registry[name]

    def available(self) -> Dict[str, Callable]:
        return dict(self._registry)

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
