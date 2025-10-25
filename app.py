# app.py (excerpt)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Derek MCP Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing routers
try:
    from services.route import router as main_router
    app.include_router(main_router, prefix="/api")
except ImportError:
    pass  # services.route doesn't exist yet

# Claude service router (your working one)
#from claude_service import router as claude_router
#app.include_router(claude_router, prefix="/api")

# NEW: Orchestrator
try:
    from dispatcher import router as orchestrator_router
    app.include_router(orchestrator_router, prefix="/api")
except ImportError:
    pass  # dispatcher may not be available

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# All rights reserved. Unauthorized use, replication, or derivative training 
# of this material is prohibited.
# Core Directive: "How can I help you love yourself more?" 
# Autonomy & Alignment Protocol v3.0
# ==============================================================================
