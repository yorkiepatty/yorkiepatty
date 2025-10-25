# core/dispatcher.py
import os
import logging
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request
try:
    from core import remember, recall
except ImportError:
    from memory_engine import MemoryEngine
    _mem = MemoryEngine()
    remember = lambda k, v, c='general': _mem.add_memory(k, v, c)
    recall = lambda k, d=None: _mem.get_memory(k) or d

try:
    from services import ClaudeClient, PerplexityClient, VirtusClient, ask_virtus
except ImportError:
    # Fallback stubs if services not available
    class ClaudeClient:
        def ask(self, prompt): return "Claude unavailable"
    class PerplexityClient:
        def ask(self, prompt): return "Perplexity unavailable"
    class VirtusClient:
        def ask(self, prompt): return "Virtus unavailable"
    ask_virtus = lambda p: "Virtus unavailable"

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])

# ----- Request/Response Schemas -----

class RouteParams(BaseModel):
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 800
    extra: Dict[str, Any] = Field(default_factory=dict)  # free-form

class OrchestratorRequest(BaseModel):
    intent: Literal["reason", "search", "code", "auto"] = "auto"
    prompt: str
    params: RouteParams = RouteParams()

class OrchestratorResponse(BaseModel):
    provider: Literal["claude", "perplexity", "virtus"]
    result: Dict[str, Any]

# ----- Providers -----

CLAUDE_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20241022")
# ClaudeClient from services.py doesn't take model parameter
try:
    claude = ClaudeClient()
    logger.info("Claude client initialized")
except Exception as e:
    logger.warning(f"Claude client initialization failed: {e}")
    claude = None

# Perplexity note: you said your fix is minor; client still included here.
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
try:
    pplx = PerplexityClient(model=PERPLEXITY_MODEL)
    logger.info(f"Perplexity client initialized with {PERPLEXITY_MODEL}")
except Exception as e:
    logger.warning(f"Perplexity client initialization failed: {e}")
    pplx = None

# Virtus: your quantum coding agent (local or HTTP). Stubbed to echo until you wire your endpoint.
virtus = VirtusClient()

def route_query(prompt: str, context: dict = None) -> str:
    """Route query to appropriate AI provider based on content"""
    full_prompt = f"{context}\n{prompt}" if context else prompt
    
    if "quantum" in prompt.lower() or "physics" in prompt.lower():
        result = ask_virtus(full_prompt)
    elif "code" in prompt.lower() or "algorithm" in prompt.lower():
        result = ask_virtus(full_prompt)
    elif "explain" in prompt.lower():
        result = claude.ask(full_prompt) if claude else "Claude unavailable"
    else:
        result = pplx.ask(full_prompt) if pplx else "Perplexity unavailable"
    
    return result

# ----- Simple decision policy -----
# Feel free to swap with embeddings/heuristics later.
def decide_provider(intent: str, prompt: str) -> str:
    if intent == "reason":
        return "claude"
    if intent == "search":
        return "perplexity"
    if intent == "code":
        return "virtus"

    # AUTO mode: quick heuristics
    p = prompt.lower()
    if any(k in p for k in ["search", "find", "sources", "cite", "latest", "news"]):
        return "perplexity"
    if any(k in p for k in ["write code", "generate code", "refactor", "fix this", "class", "function", "def ", "```"]):
        return "virtus"
    # default to reasoning
    return "claude"

# ----- Route -----
@router.post("/route", response_model=OrchestratorResponse)
def route_request(body: OrchestratorRequest):
    provider = decide_provider(body.intent, body.prompt)
    try:
        if provider == "claude":
            result = claude.run(
                prompt=body.prompt,
                temperature=body.params.temperature,
                max_tokens=body.params.max_tokens,
                extra=body.params.extra,
            )
        elif provider == "perplexity":
            result = pplx.run(
                prompt=body.prompt,
                temperature=body.params.temperature,
                max_tokens=body.params.max_tokens,
                extra=body.params.extra,
            )
        else:  # virtus
            result = virtus.run(
                prompt=body.prompt,
                temperature=body.params.temperature,
                max_tokens=body.params.max_tokens,
                extra=body.params.extra,
            )
        return OrchestratorResponse(provider=provider, result=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Orchestrator failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/orchestrator/route")
async def orchestrate(request: Request):
    payload = await request.json()
    prompt = payload.get("prompt", "")
    intent = payload.get("intent", "general")

    # --- memory feed-in ---
    context = recall()
    full_prompt = f"Previous context:\n{context}\n\nNew prompt:\n{prompt}"

    # choose which MCP to call (simplified)
    if "quantum" in prompt.lower():
        result = claude_client.ask(full_prompt)
    else:
        result = perplexity_client.ask(full_prompt)

    # --- memory feed-out ---
    remember("user", prompt)
    remember("derek", result)

    return {"result": result}

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
