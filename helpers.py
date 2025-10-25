"""Helper utilities for Derek Dashboard."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow merge of two dictionaries."""
    result = base.copy()
    result.update(overrides)
    logger.debug("Merged dictionaries into %s", result)
    return result

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
