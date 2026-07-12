# OpenJustice.ai - Document Management Service (PostgreSQL)
"""
File upload, processing, and analysis management for legal documents.

Provides:
- POST /api/documents/upload — Multipart PDF upload (max 50MB)
- GET /api/documents/{id} — Get document analysis results
- GET /api/documents — List user's uploaded documents
- DELETE /api/documents/{id} — Delete document
- PostgreSQL-backed storage via db.repository
"""

import sys
import asyncio
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from config import UPLOAD_DIR, MAX_UPLOAD_SIZE_MB, ALLOWED_UPLOAD_EXTENSIONS
from db.database import get_db
from db.repository import (
    create_document, get_document_by_id, list_documents as db_list_documents,
    update_document, delete_document as db_delete_document,
    update_user,
)

logger = logging.getLogger(__name__)

# =====================
# Pydantic Models
# =====================


class DocumentResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    content_type: str
    status: str
    user_id: str
    created_at: str
    processing_results: Optional[Dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    status: str
    message: str


# =====================
# Validation Helpers
# =====================


def validate_file_extension(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_UPLOAD_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    return file_size <= max_bytes


def _process_document_with_rag(file_path: Path, document_id: str) -> Dict[str, Any]:
    """Process a PDF document using the existing LegalDocumentProcessor."""
    try:
        from rag_pipeline.document_processor import LegalDocumentProcessor

        processor = LegalDocumentProcessor()
        result = processor.process_pdf(str(file_path), document_id=document_id)

        if result.processing_status == "failed":
            return {"status": "failed", "error": result.error_message or "Unknown processing error"}

        return {
            "status": "completed",
            "chunk_count": len(result.chunks),
            "text_preview": result.full_text[:1000] if result.full_text else "",
            "text_length": len(result.full_text),
            "metadata": result.metadata,
        }

    except ImportError as e:
        logger.warning("Document processor not available: %s", e)
        return {"status": "pending", "note": "Document queued for processing (processor unavailable)"}
    except Exception as e:
        logger.error("Document processing error for %s: %s", document_id, e)
        return {"status": "failed", "error": str(e)}


# =====================
# Router
# =====================

router = APIRouter(prefix="/api/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF document for legal analysis."""
    user_id = current_user.get("user_id")

    if not validate_file_extension(file.filename or ""):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    contents = await file.read()
    file_size = len(contents)

    if not validate_file_size(file_size):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB. Uploaded: {file_size / (1024*1024):.1f}MB",
        )

    document_id = str(uuid.uuid4())
    safe_filename = Path(file.filename or "document.pdf").name
    upload_path = Path(UPLOAD_DIR) / user_id / f"{document_id}_{safe_filename}"

    try:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(contents)

        logger.info("Saved upload %s by user %s: %s (%d bytes)", document_id, user_id, safe_filename, file_size)

        # Process document with RAG pipeline (offloaded to thread pool to avoid blocking the event loop)
        loop = asyncio.get_event_loop()
        processing_results = await loop.run_in_executor(
            None, _process_document_with_rag, upload_path, document_id,
        )
        status_str = processing_results.get("status", "processing")

        # Create document record in Postgres
        doc = await create_document(db, user_id=user_id, filename=safe_filename, status=status_str)

        # Update processing results
        if processing_results.get("status") != "pending":
            await update_document(
                db, doc.id,
                status=status_str,
                extracted_text=processing_results.get("text_preview", ""),
                classification_analysis=processing_results.get("metadata", {}).get("classification_analysis"),
            )

        # Update user's document count (fetch current count, then increment)
        from db.repository import get_user_by_id
        user_record = await get_user_by_id(db, user_id)
        current_count = user_record.documents_uploaded if user_record else 0
        await update_user(db, user_id, documents_uploaded=current_count + 1)

        return DocumentUploadResponse(
            document_id=doc.id,
            filename=safe_filename,
            size_bytes=file_size,
            status=status_str,
            message="Document uploaded successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload failed: {str(e)}")


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents uploaded by the authenticated user."""
    user_id = current_user.get("user_id")
    docs, total = await db_list_documents(db, user_id, page=page, page_size=page_size)

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                size_bytes=0,  # Not stored in DB — could be added
                content_type="application/pdf",
                status=d.status,
                user_id=d.user_id,
                created_at=d.created_at.isoformat() if d.created_at else "",
                processing_results={
                    "status": d.status,
                    "extracted_text": d.extracted_text[:200] if d.extracted_text else "",
                    "entities": d.entities or [],
                    "classification_analysis": d.classification_analysis,
                } if d.status == "completed" else None,
            )
            for d in docs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get details and processing results for a specific document."""
    user_id = current_user.get("user_id")
    doc = await get_document_by_id(db, document_id)

    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        size_bytes=0,
        content_type="application/pdf",
        status=doc.status,
        user_id=doc.user_id,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        processing_results={
            "status": doc.status,
            "text_preview": doc.extracted_text,
            "entities": doc.entities or [],
            "classification_analysis": doc.classification_analysis,
        } if doc.status == "completed" else None,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an uploaded document and its associated analysis."""
    user_id = current_user.get("user_id")
    doc = await get_document_by_id(db, document_id)

    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    await db_delete_document(db, document_id)
    logger.info("Deleted document %s for user %s", document_id, user_id)
