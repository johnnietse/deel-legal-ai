"""
Async database engine and session management for Neon PostgreSQL.

Usage:
    from db.database import get_db, init_db, close_db
    async with get_db() as session:
        ...
"""

import logging
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine
)

from config import DATABASE_URL
from db.models import Base

logger = logging.getLogger(__name__)

# ── Convert standard postgresql:// → postgresql+asyncpg:// ──
_engine = None
_async_session_factory = None


def _make_async_url(url: str) -> str:
    """Replace postgresql:// with postgresql+asyncpg:// for async driver.

    Also translates sslmode=require (libpq-style) to ssl=require (asyncpg-style).
    """
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    # asyncpg doesn't support sslmode; use ssl=require instead
    url = url.replace("?sslmode=require", "?ssl=require")
    url = url.replace("&sslmode=require", "&ssl=require")

    return url


def get_engine():
    global _engine
    if _engine is None:
        async_url = _make_async_url(DATABASE_URL)
        _engine = create_async_engine(
            async_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created.")


async def close_db():
    """Dispose of the engine (call on app shutdown)."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine disposed.")
