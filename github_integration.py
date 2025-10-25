"""GitHub integration helpers for Derek Dashboard API."""

import logging
from typing import Dict, Any

try:
    from config import Settings
except ImportError:
    # Fallback settings if config not available
    class Settings:
        GITHUB_TOKEN = None
        GITHUB_REPO = "Nathaniel-AI/DerekC"

try:
    from github import Github  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Github = None

logger = logging.getLogger(__name__)


def get_repository_metadata() -> Dict[str, Any]:
    """Fetch repository metadata using PyGithub if available."""
    settings = Settings()
    token = settings.GITHUB_TOKEN
    repo_name = settings.GITHUB_REPO

    if Github is None or not token:
        logger.warning("GitHub integration unavailable; provide GITHUB_TOKEN")
        return {"repository": repo_name, "available": False}

    client = Github(token)
    repo = client.get_repo(repo_name)
    return {
        "repository": repo.full_name,
        "stars": repo.stargazers_count,
        "open_issues": repo.open_issues_count,
    }

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
