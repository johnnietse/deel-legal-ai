# OpenJustice.ai - API Router Aggregator
"""
Aggregates all API routers into a single APIRouter for mounting on the FastAPI app.

Usage:
    from api.router import api_router
    app.include_router(api_router)
"""

import logging
from fastapi import APIRouter

# Setup logging
logger = logging.getLogger(__name__)

# Main API router
api_router = APIRouter()

# =====================
# Import and Include Routers
# =====================

# Authentication endpoints: POST /api/auth/register, /login, /token, /google
from api.auth import router as auth_router
api_router.include_router(auth_router)

# User management endpoints: GET/PATCH /api/users/me, /api/users/me/usage
from api.users import router as users_router
api_router.include_router(users_router)

# Document management endpoints: POST /api/documents/upload, GET /api/documents, etc.
from api.documents import router as documents_router
api_router.include_router(documents_router)

# Chat/streaming endpoints: GET /api/chat/stream, POST/GET /api/chat/conversations
from api.streaming import router as streaming_router
api_router.include_router(streaming_router)

# API key management endpoints: GET/POST/DELETE /api/auth/keys
from api.api_keys import router as api_keys_router
api_router.include_router(api_keys_router)

# DeepSearch + TTS endpoints: POST /rag/deepsearch, /rag/deepsearch/followup, /api/chat/tts, /api/chat/tts/voices
from api.deepsearch import router as deepsearch_router
api_router.include_router(deepsearch_router)

# Structured search endpoints: POST /rag/search
from api.search import router as search_router
api_router.include_router(search_router)


@api_router.get("/api/endpoints", tags=["System"])
async def list_endpoints():
    """
    List all available API endpoints with their methods.
    
    Useful for frontend discovery and debugging.
    """
    routes = []
    for route in api_router.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            for method in route.methods:
                if method in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                    routes.append({
                        "method": method,
                        "path": route.path,
                        "name": route.name or "",
                    })
    
    return {
        "endpoints": sorted(routes, key=lambda r: r["path"]),
        "total": len(routes),
    }
