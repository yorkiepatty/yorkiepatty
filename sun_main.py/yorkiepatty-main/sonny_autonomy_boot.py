"""Autonomous Boot Module for Sonny's Self-Improvement"""

import logging
from ai_learning_engine import get_self_improvement_engine

logger = logging.getLogger(__name__)


def main():
    """Start Sonny's autonomous learning engine."""
    logger.info("Launching Sonny's autonomous learning engine...")
    engine = get_self_improvement_engine()
    engine.start_learning()
    logger.info("Sonny's self-improvement engine is now active.")

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
