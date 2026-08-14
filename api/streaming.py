# OpenJustice.ai - Chat Streaming Service (PostgreSQL)
"""
Server-Sent Events (SSE) streaming for chat responses and conversation management.

Provides:
- GET /api/chat/stream — SSE endpoint for token-by-token RAG streaming
- POST /api/chat/conversations — Save a conversation (PostgreSQL)
- GET /api/chat/conversations — List user's conversations
- GET /api/chat/conversations/{id} — Get conversation history
"""

import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user, get_optional_user
from db.database import get_db
from db.repository import (
    create_conversation as db_create_conversation,
    get_conversation_by_id, list_conversations as db_list_conversations,
    update_conversation as db_update_conversation,
    delete_conversation as db_delete_conversation,
    add_message, get_messages, update_user,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


# =====================
# Pydantic Models
# =====================


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New Conversation", max_length=200)
    messages: List[ChatMessage] = Field(default_factory=list)


class ConversationUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    messages: Optional[List[ChatMessage]] = Field(default=None)


class ConversationResponse(BaseModel):
    id: str
    title: str
    message_count: int
    created_at: str
    updated_at: str
    user_id: str


class ConversationDetailResponse(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
    user_id: str


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse]
    total: int


# =====================
# SSE Streaming
# =====================


async def _token_generator(question: str) -> AsyncGenerator[str, None]:
    """
    Generate streaming response tokens from the RAG pipeline.
    """
    loop = asyncio.get_event_loop()

    try:
        from rag_pipeline.services import build_rag_query

        rag = build_rag_query()

        def _run_query():
            return rag.query(question=question, top_k=5)

        response = await loop.run_in_executor(None, _run_query)

        answer = response.answer
        if answer:
            words = answer.split(" ")
            chunk = ""
            for i, word in enumerate(words):
                chunk += word + " "
                if (i + 1) % 3 == 0 or i == len(words) - 1:
                    yield json.dumps({"type": "token", "content": chunk.strip()})
                    chunk = ""
                    await asyncio.sleep(0.05)
        else:
            yield json.dumps({
                "type": "token",
                "content": "I could not find relevant information to answer your question.",
            })

        sources_data = []
        for src in response.sources:
            sources_data.append({
                "index": src.get("index", 0),
                "case_name": src.get("case_name", "Unknown"),
                "citation": src.get("citation", ""),
                "excerpt": src.get("excerpt", "")[:300],
                "score": src.get("score", 0.0),
                "jurisdiction": src.get("jurisdiction", ""),
                "court": src.get("court", ""),
            })

        yield json.dumps({
            "type": "sources",
            "content": {
                "count": len(sources_data),
                "sources": sources_data,
                "confidence": response.confidence,
                "retrieval_mode": getattr(response, "retrieval_mode", "hybrid"),
            },
        })

        yield json.dumps({"type": "done", "content": {}})

    except ImportError as e:
        logger.error("RAG pipeline import failed: %s", e)
        yield json.dumps({"type": "error", "content": "RAG service is not available. Please contact support."})
    except Exception as e:
        logger.error("Streaming error: %s", e)
        yield json.dumps({"type": "error", "content": f"An error occurred: {str(e)}"})


@router.get("/api/chat/stream")
async def stream_chat(
    request: Request,
    question: str = Query(..., min_length=3, max_length=5000, description="Legal question to answer"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
):
    """
    Stream a RAG-powered legal answer via Server-Sent Events.
    """
    import anyio
    from sse_starlette import EventSourceResponse

    shutdown_event = anyio.Event()

    async def event_generator():
        try:
            async for token_data in _token_generator(question):
                if await request.is_disconnected():
                    logger.info("Client disconnected from SSE stream")
                    break
                yield {"data": token_data}
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("SSE generator error: %s", e)
            yield {"data": json.dumps({"type": "error", "content": "Stream interrupted"})}

    return EventSourceResponse(
        event_generator(),
        ping=30,
        send_timeout=60,
        shutdown_event=shutdown_event,
        shutdown_grace_period=5.0,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# =====================
# Conversation CRUD (PostgreSQL)
# =====================


@router.post("/api/chat/conversations", response_model=ConversationDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: ConversationCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save a new conversation."""
    user_id = current_user.get("user_id")

    conv = await db_create_conversation(db, user_id=user_id, title=request.title or "New Conversation")

    # Add initial messages
    for msg in request.messages:
        await add_message(
            db,
            conversation_id=conv.id,
            role=msg.role,
            content=msg.content,
        )

    # Update user conversation count
    from sqlalchemy import select
    from db.models import User
    result = await db.execute(select(User).where(User.id == user_id))
    user_row = result.scalar_one_or_none()
    if user_row:
        await update_user(db, user_id, conversations_count=user_row.conversations_count + 1)

    messages = await get_messages(db, conv.id)

    logger.info("Created conversation %s for user %s", conv.id, user_id)

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        messages=[ChatMessage(role=m.role, content=m.content, timestamp=m.created_at.isoformat()) for m in messages],
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
        user_id=conv.user_id,
    )


@router.get("/api/chat/conversations", response_model=ConversationListResponse)
async def list_conversations(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all conversations for the authenticated user."""
    user_id = current_user.get("user_id")
    conversations = await db_list_conversations(db, user_id)

    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.id,
                title=c.title,
                message_count=0,  # Could be optimized with a count query
                created_at=c.created_at.isoformat() if c.created_at else "",
                updated_at=c.updated_at.isoformat() if c.updated_at else "",
                user_id=c.user_id,
            )
            for c in conversations
        ],
        total=len(conversations),
    )


@router.get("/api/chat/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a full conversation with all messages."""
    user_id = current_user.get("user_id")
    conv = await get_conversation_by_id(db, conversation_id)

    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if conv.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    messages = await get_messages(db, conversation_id)

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        messages=[ChatMessage(role=m.role, content=m.content, timestamp=m.created_at.isoformat()) for m in messages],
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
        user_id=conv.user_id,
    )


@router.patch("/api/chat/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a conversation (title and/or messages)."""
    user_id = current_user.get("user_id")
    conv = await get_conversation_by_id(db, conversation_id)

    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if conv.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if request.title is not None:
        await db_update_conversation(db, conversation_id, title=request.title)

    if request.messages is not None:
        # Delete existing messages and re-add
        from sqlalchemy import delete
        from db.models import Message
        await db.execute(delete(Message).where(Message.conversation_id == conversation_id))
        for msg in request.messages:
            await add_message(db, conversation_id=conversation_id, role=msg.role, content=msg.content)

    messages = await get_messages(db, conversation_id)
    conv = await get_conversation_by_id(db, conversation_id)

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        messages=[ChatMessage(role=m.role, content=m.content, timestamp=m.created_at.isoformat()) for m in messages],
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        updated_at=conv.updated_at.isoformat() if conv.updated_at else "",
        user_id=conv.user_id,
    )


@router.delete("/api/chat/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation."""
    user_id = current_user.get("user_id")
    conv = await get_conversation_by_id(db, conversation_id)

    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if conv.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    await db_delete_conversation(db, conversation_id)
    logger.info("Deleted conversation %s for user %s", conversation_id, user_id)
