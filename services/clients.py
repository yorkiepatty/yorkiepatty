# services/clients.py
import os
import logging
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class ClaudeClient:
    def __init__(self, model: str):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.model = model
        self.url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def run(self, prompt: str, temperature: Optional[float], max_tokens: Optional[int], extra: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or 800,
            "temperature": temperature if temperature is not None else 0.7,
            "messages": [{"role": "user", "content": prompt}],
        }
        if extra:
            payload.update(extra)
        r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        if r.status_code != 200:
            logger.error(f"Claude error {r.status_code}: {r.text}")
            raise RuntimeError(r.text)
        return r.json()

class PerplexityClient:
    def __init__(self, model: str):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise RuntimeError("PERPLEXITY_API_KEY not set")
        self.model = model
        self.url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def run(self, prompt: str, temperature: Optional[float], max_tokens: Optional[int], extra: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or 800,
            "temperature": temperature if temperature is not None else 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        if extra:
            payload.update(extra)
        r = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        if r.status_code not in (200, 201):
            logger.error(f"Perplexity error {r.status_code}: {r.text}")
            raise RuntimeError(r.text)
        return r.json()

class VirtusClient:
    """
    Wire this to your Virtus engine.
    For now this is a placeholder that echoes the plan.
    Replace `self.url` + POST call when your Virtus service endpoint is ready.
    """
    def __init__(self):
        self.url = os.getenv("VIRTUS_URL", "").strip()

    def run(self, prompt: str, temperature: Optional[float], max_tokens: Optional[int], extra: Dict[str, Any]) -> Dict[str, Any]:
        if not self.url:
            # Local stub
            return {
                "virtus": "stub",
                "plan": [
                    "Analyze problem",
                    "Generate or repair code",
                    "Explain diffs and rationale"
                ],
                "prompt": prompt,
            }
        # Example when you have an endpoint:
        # r = requests.post(self.url, json={"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, **extra}, timeout=120)
        # if r.status_code != 200:
        #     raise RuntimeError(r.text)
        # return r.json()
        return {"error": "Virtus URL configured but handler not implemented yet"}


# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
