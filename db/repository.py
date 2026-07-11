"""
Data access layer for OpenJustice.ai.

Provides async CRUD operations for all models,
replacing the previous JSON file persistence.
"""

import uuid
import logging
import secrets
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Tuple

from sqlalchemy import select, func, delete, and_, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User, Document, Conversation, Message, Feedback, ApiKey

logger = logging.getLogger(__name__)


def generate_api_key_value() -> tuple[str, str]:
    """Generate a new API key pair: (plaintext, sha256_hash)."""
    random_part = secrets.token_urlsafe(48)
    plaintext = f"oj_{random_part}"
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return plaintext, key_hash


# ═══════════════════════════════════════════
#  Users
# ═══════════════════════════════════════════

async def create_user(
    db: AsyncSession,
    email: str,
    name: str,
    password_hash: Optional[str] = None,
    google_id: Optional[str] = None,
    tier: str = "free",
) -> User:
    user = User(
        id=str(uuid.uuid4()),
        email=email,
        name=name,
        password_hash=password_hash,
        google_id=google_id,
        tier=tier,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info("Created user %s (%s)", user.id, email)
    return user


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_google_id(db: AsyncSession, google_id: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.google_id == google_id))
    return result.scalar_one_or_none()


async def update_user(db: AsyncSession, user_id: str, **kwargs) -> Optional[User]:
    user = await get_user_by_id(db, user_id)
    if not user:
        return None
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    user.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)
    return user


async def increment_query_count(db: AsyncSession, user_id: str) -> Optional[User]:
    return await update_user(db, user_id, queries_used=User.queries_used + 1)


async def get_user_count(db: AsyncSession) -> int:
    result = await db.execute(select(func.count(User.id)))
    return result.scalar() or 0


# ═══════════════════════════════════════════
#  Documents
# ═══════════════════════════════════════════

async def create_document(
    db: AsyncSession,
    user_id: str,
    filename: str,
    status: str = "processing",
) -> Document:
    doc = Document(
        id=str(uuid.uuid4()),
        user_id=user_id,
        filename=filename,
        status=status,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def get_document_by_id(db: AsyncSession, doc_id: str) -> Optional[Document]:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def list_documents(
    db: AsyncSession,
    user_id: str,
    page: int = 1,
    page_size: int = 20,
) -> Tuple[List[Document], int]:
    count_q = select(func.count(Document.id)).where(Document.user_id == user_id)
    total = (await db.execute(count_q)).scalar() or 0

    q = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(q)
    return list(result.scalars().all()), total


async def update_document(
    db: AsyncSession, doc_id: str, **kwargs
) -> Optional[Document]:
    doc = await get_document_by_id(db, doc_id)
    if not doc:
        return None
    for key, value in kwargs.items():
        if hasattr(doc, key):
            setattr(doc, key, value)
    doc.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def delete_document(db: AsyncSession, doc_id: str) -> bool:
    doc = await get_document_by_id(db, doc_id)
    if not doc:
        return False
    await db.delete(doc)
    await db.commit()
    return True


# ═══════════════════════════════════════════
#  Conversations & Messages
# ═══════════════════════════════════════════

async def create_conversation(
    db: AsyncSession,
    user_id: str,
    title: str = "New Conversation",
) -> Conversation:
    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=title,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def get_conversation_by_id(db: AsyncSession, conv_id: str) -> Optional[Conversation]:
    result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
    return result.scalar_one_or_none()


async def list_conversations(
    db: AsyncSession, user_id: str
) -> List[Conversation]:
    q = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
    )
    result = await db.execute(q)
    return list(result.scalars().all())


async def update_conversation(
    db: AsyncSession, conv_id: str, **kwargs
) -> Optional[Conversation]:
    conv = await get_conversation_by_id(db, conv_id)
    if not conv:
        return None
    for key, value in kwargs.items():
        if hasattr(conv, key):
            setattr(conv, key, value)
    conv.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(conv)
    return conv


async def delete_conversation(db: AsyncSession, conv_id: str) -> bool:
    conv = await get_conversation_by_id(db, conv_id)
    if not conv:
        return False
    await db.delete(conv)
    await db.commit()
    return True


async def add_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    sources: Optional[dict] = None,
) -> Message:
    msg = Message(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources,
        created_at=datetime.now(timezone.utc),
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_messages(
    db: AsyncSession, conversation_id: str
) -> List[Message]:
    q = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    result = await db.execute(q)
    return list(result.scalars().all())


# ═══════════════════════════════════════════
#  Feedback
# ═══════════════════════════════════════════

async def create_feedback(
    db: AsyncSession,
    query: str,
    user_id: Optional[str] = None,
    response: Optional[str] = None,
    rating: Optional[int] = None,
    feedback_text: Optional[str] = None,
    metadata_json: Optional[dict] = None,
) -> Feedback:
    fb = Feedback(
        id=str(uuid.uuid4()),
        user_id=user_id,
        query=query,
        response=response,
        rating=rating,
        feedback_text=feedback_text,
        metadata_json=metadata_json,
        created_at=datetime.now(timezone.utc),
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return fb


async def get_feedback_summary(db: AsyncSession) -> dict:
    total = (await db.execute(select(func.count(Feedback.id)))).scalar() or 0
    avg_rating = (await db.execute(select(func.avg(Feedback.rating)))).scalar()
    return {
        "total_feedback": total,
        "average_rating": round(float(avg_rating), 2) if avg_rating else None,
    }


async def list_feedback(db: AsyncSession, limit: int = 50) -> List[Feedback]:
    q = select(Feedback).order_by(Feedback.created_at.desc()).limit(limit)
    result = await db.execute(q)
    return list(result.scalars().all())


# ═══════════════════════════════════════════
#  API Keys
# ═══════════════════════════════════════════

async def create_api_key(
    db: AsyncSession,
    user_id: str,
    name: str,
    key_hash: str,
) -> ApiKey:
    ak = ApiKey(
        id=str(uuid.uuid4()),
        user_id=user_id,
        name=name,
        key_hash=key_hash,
        created_at=datetime.now(timezone.utc),
    )
    db.add(ak)
    await db.commit()
    await db.refresh(ak)
    return ak


async def list_api_keys(db: AsyncSession, user_id: str) -> List[ApiKey]:
    q = select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
    result = await db.execute(q)
    return list(result.scalars().all())


async def delete_api_key(db: AsyncSession, key_id: str) -> bool:
    ak = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    ak = ak.scalar_one_or_none()
    if not ak:
        return False
    await db.delete(ak)
    await db.commit()
    return True


async def get_api_key_by_hash(db: AsyncSession, key_hash: str) -> Optional[ApiKey]:
    stmt = select(ApiKey).where(ApiKey.key_hash == key_hash)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def update_api_key_last_used(db: AsyncSession, key_id: str) -> None:
    stmt = update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=datetime.now(timezone.utc))
    await db.execute(stmt)
    await db.commit()
