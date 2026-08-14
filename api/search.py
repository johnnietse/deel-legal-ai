"""Structured search endpoint for the Search UI."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List

from api.auth import get_current_user, get_api_key_user

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])

class SearchResult(BaseModel):
    id: str
    title: str
    excerpt: str
    url: str = ""
    source_type: str
    jurisdiction: str = ""
    court: str = ""
    year: str = ""
    citation: str = ""
    relevance_score: float = 0.0


@router.post("/rag/search")
async def search_endpoint(
    query: str = Query(..., min_length=1, max_length=2000),
    jurisdiction: Optional[str] = Query(None),
    source_type: Optional[str] = Query(None, pattern="^(case_law|web|statute|bm25)$"),
    year_from: Optional[int] = Query(None, ge=1900, le=2030),
    year_to: Optional[int] = Query(None, ge=1900, le=2030),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    sort_by: str = Query("relevance", pattern="^(relevance|date)$"),
    user=Depends(get_current_user),
    api_user=Depends(get_api_key_user),
):
    """Structured search across the legal corpus. Returns results, not chat."""
    effective_user = user or api_user
    if not effective_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Search vector store with filters
    from rag_pipeline.vector_store import create_vector_store
    from rag_pipeline.legal_document_ingester import generate_embedding
    from rag_pipeline.gemini_key_manager import search_key_manager
    from config import VECTOR_STORE_BACKEND, CHUNK_NAMESPACE

    store = create_vector_store(backend=VECTOR_STORE_BACKEND)
    filter_dict = {}
    if jurisdiction:
        filter_dict["jurisdiction"] = jurisdiction
    if year_from:
        filter_dict["year"] = {"$gte": str(year_from)}

    # Embed the query into a vector — the store requires query_vector, not raw text.
    # Route through the dedicated search key manager so ingestion never starves search.
    query_vector = generate_embedding(query, km=search_key_manager)
    results = store.search(
        query_vector=query_vector,
        top_k=page_size * page,
        namespace=CHUNK_NAMESPACE,
        filter=filter_dict if filter_dict else None,
    )

    # Paginate
    start = (page - 1) * page_size
    page_results = results[start:start + page_size]

    def _norm(r):
        if isinstance(r, dict):
            return r.get("id", ""), r.get("metadata", {}), r.get("score", 0.0)
        return r.id, r.metadata, r.score

    return {
        "results": [
            SearchResult(
                id=rid,
                title=meta.get("title", ""),
                excerpt=(meta.get("content", "") or "")[:500],
                source_type="case_law",
                jurisdiction=str(meta.get("jurisdiction", "") or ""),
                court=str(meta.get("court", "") or ""),
                year=str(meta.get("year", "") or ""),
                citation=str(meta.get("citation", "") or ""),
                relevance_score=score,
            )
            for r in page_results
            for rid, meta, score in [_norm(r)]
        ],
        "total": len(results),
        "page": page,
        "page_size": page_size,
    }
