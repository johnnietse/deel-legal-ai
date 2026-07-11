# OpenJustice.ai - Authentication Service
"""
JWT-based authentication with password hashing and Google OAuth.

Provides:
- JWT access + refresh token generation and validation
- Password hashing with passlib[bcrypt] (CryptContext)
- Google OAuth via google-auth library
- Bearer token dependency injection for route protection
- Postgres-backed user lookups via db.repository
- API key authentication via X-API-Key header
"""

import sys
import uuid
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    JWT_SECRET_KEY, JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES, JWT_REFRESH_TOKEN_EXPIRE_DAYS,
    GOOGLE_OAUTH_CLIENT_ID,
)
from db.database import get_db
from db.repository import (
    get_user_by_id, get_user_by_email, get_user_by_google_id,
    create_user, update_user, get_api_key_by_hash, update_api_key_last_used,
)

logger = logging.getLogger(__name__)

# =====================
# Security Setup
# =====================

# bcrypt settings
BCRYPT_ROUNDS = 12

security = HTTPBearer(auto_error=False)


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()

# =====================
# Pydantic Models
# =====================


class TokenResponse(BaseModel):
    """JWT token pair response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(default=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60)


class TokenRefreshRequest(BaseModel):
    """Refresh token request body"""
    refresh_token: str


class GoogleOAuthRequest(BaseModel):
    """Google OAuth ID token from frontend"""
    id_token: str
    client_id: Optional[str] = None


class AuthResponse(BaseModel):
    """Authentication response with user info"""
    user_id: str
    email: str
    name: str
    tier: str
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# =====================
# Password Helpers
# =====================


def hash_password(password: str) -> str:
    """Hash a password using bcrypt directly."""
    salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# =====================
# JWT Helpers
# =====================


def create_access_token(
    user_id: str,
    email: str,
    tier: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    now = datetime.now(timezone.utc)
    expires = now + (expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES))

    claims = {
        "sub": user_id,
        "email": email,
        "tier": tier,
        "type": "access",
        "iat": now,
        "exp": expires,
        "jti": str(uuid.uuid4()),
    }

    return jwt.encode(claims, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def create_refresh_token(
    user_id: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    now = datetime.now(timezone.utc)
    expires = now + (expires_delta or timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS))

    claims = {
        "sub": user_id,
        "type": "refresh",
        "iat": now,
        "exp": expires,
        "jti": str(uuid.uuid4()),
    }

    return jwt.encode(claims, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_signature": True, "verify_exp": True, "require_exp": True},
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTClaimsError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token claims: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =====================
# Dependencies
# =====================


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency that extracts and validates the current user from JWT.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Use an access token.",
        )

    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "tier": payload.get("tier", "free"),
        "token_jti": payload.get("jti"),
    }


async def get_optional_user(
    request: Request,
) -> Optional[Dict[str, Any]]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:]
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            return None
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "tier": payload.get("tier", "free"),
        }
    except (HTTPException, JWTError):
        return None


async def get_api_key_user(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Optional[Dict[str, Any]]:
    """Authenticate via API key header. Returns user dict or None."""
    if not x_api_key:
        return None
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    api_key = await get_api_key_by_hash(db, key_hash)
    if not api_key:
        return None
    # Update last_used_at
    await update_api_key_last_used(db, api_key.id)
    user = await get_user_by_id(db, api_key.user_id)
    if not user:
        return None
    return {"user_id": user.id, "email": user.email, "tier": user.tier, "auth_method": "api_key"}


# =====================
# Google OAuth
# =====================


async def verify_google_id_token(id_token: str, expected_client_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify a Google ID token and extract user information.
    Uses Google's google-auth library to verify the token signature.
    """
    client_id = expected_client_id or GOOGLE_OAUTH_CLIENT_ID
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth is not configured. Set GOOGLE_OAUTH_CLIENT_ID in environment.",
        )

    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        id_info = google_id_token.verify_oauth2_token(
            id_token,
            google_requests.Request(),
            audience=client_id,
        )

        if id_info.get("iss") not in ["accounts.google.com", "https://accounts.google.com"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer")

        email = id_info.get("email", "")
        name = id_info.get("name", "")
        google_sub = id_info.get("sub", "")
        picture = id_info.get("picture", "")

        if not email:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email not found in token")

        logger.info("Google OAuth token verified for %s (sub=%s...)", email, google_sub[:8] if google_sub else "?")

        return {
            "email": email,
            "name": name or email.split("@")[0],
            "google_sub": google_sub,
            "picture": picture,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid Google ID token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google ID token")


# =====================
# Router
# =====================

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/token", response_model=TokenResponse)
async def create_token_from_refresh(request: TokenRefreshRequest):
    """
    Get a new access token using a refresh token.
    """
    try:
        payload = decode_token(request.refresh_token)

        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type. Expected refresh token.")

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        access_token = create_access_token(user_id=user_id, email=payload.get("email", ""), tier=payload.get("tier", "free"))
        new_refresh_token = create_refresh_token(user_id=user_id)

        return TokenResponse(access_token=access_token, refresh_token=new_refresh_token)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token refresh failed")


@router.post("/google", response_model=AuthResponse)
async def google_oauth_login(
    request: GoogleOAuthRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Log in or register using a Google ID token.
    """
    try:
        google_user = await verify_google_id_token(
            request.id_token,
            expected_client_id=request.client_id,
        )

        email = google_user["email"]
        google_sub = google_user["google_sub"]
        name = google_user["name"]

        # Try to find existing user by google_id or email
        user = await get_user_by_google_id(db, google_sub)
        if not user:
            user = await get_user_by_email(db, email)

        if user:
            # Link google_id if not set
            if not user.google_id:
                user = await update_user(db, user.id, google_id=google_sub)
        else:
            # Create new user
            user = await create_user(
                db,
                email=email,
                name=name,
                google_id=google_sub,
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
        logger.error("Google OAuth error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Google authentication failed")
