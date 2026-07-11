"""API key management endpoints."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.repository import (
    create_api_key, list_api_keys, delete_api_key,
    generate_api_key_value, get_api_key_by_hash,
)
from api.auth import get_current_user, hash_api_key
from db.models import ApiKey

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["api-keys"])


@router.get("/keys")
async def get_keys(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's API keys (masked)."""
    keys = await list_api_keys(db, user["user_id"])
    return {
        "keys": [
            {
                "id": k.id,
                "name": k.name,
                "key_preview": k.key_hash[:12] + "...",
                "created_at": k.created_at.isoformat() if k.created_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
            }
            for k in keys
        ]
    }


@router.post("/keys")
async def create_key_route(
    name: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a new API key."""
    plaintext, key_hash = generate_api_key_value()
    api_key = await create_api_key(db, user["user_id"], name, key_hash)
    return {
        "id": api_key.id,
        "name": api_key.name,
        "key": plaintext,  # Only returned once!
        "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
    }


@router.delete("/keys/{key_id}")
async def revoke_key_route(
    key_id: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke an API key."""
    # Verify ownership
    stmt = select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user["user_id"])
    result = await db.execute(stmt)
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    deleted = await delete_api_key(db, key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"message": "Key revoked"}