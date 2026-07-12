# API Package
"""
FastAPI service for Deel Lab Legal AI System / OpenJustice.ai

Submodules:
- main: Core FastAPI app with existing endpoints
- auth: JWT authentication, password hashing, Google OAuth
- users: User registration, login, profile, usage tracking
- documents: File upload, PDF processing, document management
- streaming: SSE chat streaming, conversation management
- middleware: Rate limiting middleware
- router: Aggregated API router for all new endpoints
"""

from api.main import app, start_server

# Export all routers for external use
from api.auth import router as auth_router
from api.users import router as users_router
from api.documents import router as documents_router
from api.streaming import router as streaming_router
from api.router import api_router
from api.middleware import RateLimitMiddleware

__all__ = [
    "app", "start_server",
    "auth_router", "users_router",
    "documents_router", "streaming_router",
    "api_router", "RateLimitMiddleware",
]
__version__ = "2.0.0"
