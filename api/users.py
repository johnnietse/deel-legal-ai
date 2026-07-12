# OpenJustice.ai - User Management Service (PostgreSQL)
"""
User registration, authentication, profile management, and usage tracking.

Provides:
- Register endpoint (POST /api/auth/register) — async, backed by db.repository
- Login endpoint (POST /api/auth/login)
- Profile GET/PATCH (GET/PATCH /api/users/me)
- Usage statistics (GET /api/users/me/usage)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field, EmailStr, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from api.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    get_current_user,
    AuthResponse,
)
from db.database import get_db
from db.repository import (
    get_user_by_id, get_user_by_email, create_user, update_user,
    list_conversations, list_documents,
)

logger = logging.getLogger(__name__)


# =====================
# Pydantic Models
# =====================


class RegisterRequest(BaseModel):
    """User registration request body"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=100)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    """User login request body"""
    email: EmailStr
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    """Public user profile response"""
    id: str
    email: str
    name: str
    tier: str
    queries_used: int
    queries_limit: int
    created_at: str
    updated_at: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    """Profile update request body (all fields optional)"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    current_password: Optional[str] = Field(default=None, min_length=1)
    new_password: Optional[str] = Field(default=None, min_length=8, max_length=128)


class UsageResponse(BaseModel):
    """User usage statistics response"""
    queries_used: int
    queries_limit: int
    queries_remaining: int
    documents_uploaded: int
    conversations_count: int
    tier: str
    period_start: str
    period_end: str


# =====================
# Constants
# =====================

TIER_LIMITS = {
    "free": {"queries_limit": 20, "price": "$0"},
    "pro": {"queries_limit": 200, "price": "$29/mo"},
    "enterprise": {"queries_limit": 999999, "price": "Custom"},
}


# =====================
# Helpers
# =====================

def _user_to_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        tier=user.tier,
        queries_used=user.queries_used,
        queries_limit=user.queries_limit,
        created_at=user.created_at.isoformat() if user.created_at else "",
        updated_at=user.updated_at.isoformat() if user.updated_at else None,
    )


# =====================
# Router
# =====================

router = APIRouter(tags=["Users"])


@router.post(
    "/api/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user account."""
    try:
        existing = await get_user_by_email(db, request.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Email '{request.email}' is already registered",
            )

        password_hash = hash_password(request.password)
        user = await create_user(
            db,
            email=request.email,
            name=request.name,
            password_hash=password_hash,
            tier="free",
        )

        access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            tier=user.tier,
        )
        refresh_token = create_refresh_token(user_id=user.id)

        return AuthResponse(
            user_id=user.id,
            email=user.email,
            name=user.name,
            tier=user.tier,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed")


@router.post("/api/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate and receive JWT tokens."""
    try:
        user = await get_user_by_email(db, request.email)
        if not user or not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        access_token = create_access_token(
            user_id=user.id,
            email=user.email,
            tier=user.tier,
        )
        refresh_token = create_refresh_token(user_id=user.id)

        return AuthResponse(
            user_id=user.id,
            email=user.email,
            name=user.name,
            tier=user.tier,
            access_token=access_token,
            refresh_token=refresh_token,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed")


@router.get("/api/users/me", response_model=UserResponse)
async def get_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the authenticated user's profile."""
    user_id = current_user.get("user_id")
    user = await get_user_by_id(db, user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return _user_to_response(user)


@router.patch("/api/users/me", response_model=UserResponse)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the authenticated user's profile."""
    try:
        user_id = current_user.get("user_id")
        user = await get_user_by_id(db, user_id)

        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        update_kwargs = {}

        if request.name is not None:
            update_kwargs["name"] = request.name.strip()

        if request.new_password is not None:
            if not request.current_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is required to set a new password",
                )
            if not user.password_hash or not verify_password(request.current_password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Current password is incorrect",
                )
            update_kwargs["password_hash"] = hash_password(request.new_password)

        if update_kwargs:
            user = await update_user(db, user_id, **update_kwargs)

        return _user_to_response(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profile update error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Profile update failed")


@router.get("/api/users/me/usage", response_model=UsageResponse)
async def get_usage_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the authenticated user's usage statistics."""
    user_id = current_user.get("user_id")
    user = await get_user_by_id(db, user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    now = datetime.now(timezone.utc)
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

    if now.month == 12:
        period_end = now.replace(year=now.year + 1, month=1, day=1)
    else:
        period_end = now.replace(month=now.month + 1, day=1)
    period_end = period_end.isoformat()

    return UsageResponse(
        queries_used=user.queries_used,
        queries_limit=user.queries_limit,
        queries_remaining=max(0, user.queries_limit - user.queries_used),
        documents_uploaded=user.documents_uploaded,
        conversations_count=user.conversations_count,
        tier=user.tier,
        period_start=period_start,
        period_end=period_end,
    )


@router.get("/api/users/me/usage/chart")
async def get_usage_chart(
    days: int = Query(30, ge=1, le=365),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return daily query usage for the last N days."""
    user_id = current_user.get("user_id")
    since = datetime.now(timezone.utc) - timedelta(days=days)

    stmt = text("""
        SELECT DATE(m.created_at) as day, COUNT(*) as count
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.user_id = :uid AND m.role = 'user' AND m.created_at >= :since
        GROUP BY DATE(m.created_at)
        ORDER BY day
    """)
    result = await db.execute(stmt, {"uid": user_id, "since": since})
    rows = result.fetchall()

    return {
        "data": [
            {"date": str(r.day), "queries": r.count, "classifications": 0}
            for r in rows
        ],
        "total_queries": sum(r.count for r in rows),
    }


@router.get("/api/users/me/activity")
async def get_recent_activity(
    limit: int = Query(10, ge=1, le=50),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return recent user activity (conversations + document uploads)."""
    user_id = current_user.get("user_id")

    convos = await list_conversations(db, user_id)
    docs, _ = await list_documents(db, user_id, page=1, page_size=limit)

    activity = []
    for c in convos:
        activity.append({
            "id": c.id,
            "type": "query",
            "description": c.title,
            "timestamp": c.created_at.isoformat() if c.created_at else None,
        })
    for d in docs:
        activity.append({
            "id": d.id,
            "type": "document",
            "description": d.filename,
            "timestamp": d.created_at.isoformat() if d.created_at else None,
        })

    activity.sort(key=lambda a: a.get("timestamp") or "", reverse=True)
    return {"data": activity[:limit]}


@router.post("/api/subscriptions/upgrade")
async def upgrade_subscription(
    tier: str = Query(..., pattern="^(pro|enterprise)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upgrade user tier. Payment integration is future work."""
    limits = {"pro": 200, "enterprise": 999999}
    if tier not in limits:
        raise HTTPException(status_code=400, detail="Invalid tier")

    updated = await update_user(db, current_user.get("user_id"), tier=tier, queries_limit=limits[tier])
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")

    return {"success": True, "tier": tier}
