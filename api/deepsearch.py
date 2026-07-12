"""DeepSearch and TTS API endpoints."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional

from api.auth import get_current_user, get_api_key_user
from rag_pipeline.deep_search import DeepSearchEngine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["deepsearch"])

class DeepSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=5000)
    max_sources: int = Field(default=15, ge=1, le=50)

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="en-CA-LiamNeural")


@router.post("/rag/deepsearch")
async def deep_search_endpoint(
    req: DeepSearchRequest,
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Multi-source deep research across case law, web, and statutes."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    engine = DeepSearchEngine()
    result = await engine.deep_search(
        query=req.query,
        user_id=effective_user.get("user_id"),
        max_sources=req.max_sources,
    )

    if result.error:
        raise HTTPException(status_code=400, detail=result.error)

    return {
        "answer": result.answer,
        "sources": [
            {
                "id": s.id,
                "title": s.title,
                "excerpt": s.excerpt,
                "url": s.url,
                "source_type": s.source_type,
                "relevance_score": round(s.relevance_score, 3),
            }
            for s in result.sources
        ],
        "source_type_counts": result.source_type_counts,
        "suggested_follow_ups": result.suggested_follow_ups,
        "processing_time_ms": result.processing_time_ms,
    }


@router.post("/rag/deepsearch/followup")
async def deep_search_followup(
    original_query: str = Query(...),
    follow_up: str = Query(...),
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Ask a follow-up question on a previous DeepSearch result."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Combine original context with follow-up
    combined = f"Original question: {original_query}\nFollow-up: {follow_up}"
    engine = DeepSearchEngine()
    result = await engine.deep_search(
        query=combined,
        user_id=effective_user.get("user_id"),
    )

    if result.error:
        raise HTTPException(status_code=400, detail=result.error)

    return {
        "answer": result.answer,
        "sources": [
            {
                "id": s.id,
                "title": s.title,
                "excerpt": s.excerpt,
                "url": s.url,
                "source_type": s.source_type,
                "relevance_score": round(s.relevance_score, 3),
            }
            for s in result.sources
        ],
        "suggested_follow_ups": result.suggested_follow_ups,
    }


@router.post("/api/chat/tts")
async def text_to_speech_endpoint(
    req: TTSRequest,
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Generate audio from text using free Microsoft TTS."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    from rag_pipeline.tts import generate_audio, AVAILABLE_VOICES

    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Available: {list(AVAILABLE_VOICES.keys())}",
        )

    audio_bytes = await generate_audio(req.text, voice=req.voice)
    if audio_bytes is None:
        raise HTTPException(status_code=503, detail="Audio generation unavailable")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename=openjustice_audio.mp3",
            "Content-Length": str(len(audio_bytes)),
        },
    )


@router.get("/api/chat/tts/voices")
async def list_voices():
    """List available TTS voices."""
    from rag_pipeline.tts import AVAILABLE_VOICES
    return {
        "voices": [
            {"id": k, "description": v}
            for k, v in AVAILABLE_VOICES.items()
        ]
    }
