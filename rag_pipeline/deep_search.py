"""DeepSearch — multi-source legal research orchestrator.

Queries Pinecone (vector), Elasticsearch (BM25), SearXNG (web),
and legislation DB in parallel, then fuses results and synthesizes
a cited answer via Gemini.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal
import asyncio

logger = logging.getLogger(__name__)


SourceType = Literal["case_law", "web", "statute", "bm25"]


@dataclass
class UnifiedSource:
    """A single source result from any backend, unified format."""
    id: str
    title: str
    excerpt: str
    url: str = ""
    source_type: SourceType = "case_law"
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class DeepSearchResult:
    """Complete DeepSearch result."""
    answer: str
    sources: List[UnifiedSource] = field(default_factory=list)
    source_type_counts: Dict[str, int] = field(default_factory=dict)
    suggested_follow_ups: List[str] = field(default_factory=list)
    processing_time_ms: int = 0
    error: Optional[str] = None


def deduplicate_sources(sources: List[UnifiedSource]) -> List[UnifiedSource]:
    """Remove duplicate sources by URL (prefer higher score)."""
    seen_urls = set()
    deduped = []
    for s in sorted(sources, key=lambda x: x.relevance_score, reverse=True):
        key = s.url or s.id
        if key not in seen_urls:
            seen_urls.add(key)
            deduped.append(s)
    return deduped


def fuse_and_rank_results(
    case_sources: List[UnifiedSource],
    web_sources: List[UnifiedSource],
    statute_sources: List[UnifiedSource],
    top_k: int = 15,
) -> List[UnifiedSource]:
    """Fuse all sources, deduplicate, and rank by relevance."""
    all_sources = case_sources + web_sources + statute_sources
    all_sources = deduplicate_sources(all_sources)
    all_sources.sort(key=lambda s: s.relevance_score, reverse=True)
    return all_sources[:top_k]


def count_source_types(sources: List[UnifiedSource]) -> Dict[str, int]:
    """Count sources by type."""
    counts = {}
    for s in sources:
        counts[s.source_type] = counts.get(s.source_type, 0) + 1
    return counts


class DeepSearchEngine:
    """Orchestrates multi-source legal deep research."""

    def __init__(self):
        from rag_pipeline.services import build_rag_query
        self.rag_query = build_rag_query()
        self.chat = self.rag_query.chat

    async def _search_vector(self, query: str, top_k: int = 10) -> List[UnifiedSource]:
        """Search Pinecone vector store."""
        try:
            from rag_pipeline.vector_store import create_vector_store
            from rag_pipeline.legal_document_ingester import generate_embedding
            from rag_pipeline.gemini_key_manager import search_key_manager
            from config import VECTOR_STORE_BACKEND, CHUNK_NAMESPACE
            store = create_vector_store(backend=VECTOR_STORE_BACKEND)
            query_vector = generate_embedding(query, km=search_key_manager)
            results = await asyncio.to_thread(
                store.search, query_vector, top_k=top_k, namespace=CHUNK_NAMESPACE
            )
            sources = []
            for r in results:
                sources.append(UnifiedSource(
                    id=r.id,
                    title=r.metadata.get("title", "Unknown Case"),
                    excerpt=(r.metadata.get("content", "") or "")[:500],
                    source_type="case_law",
                    relevance_score=r.score,
                    metadata=r.metadata,
                ))
            return sources
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def _search_bm25(self, query: str, top_k: int = 10) -> List[UnifiedSource]:
        """Search Elasticsearch BM25."""
        try:
            from config import ELASTICSEARCH_URL
            import requests
            es_query = {
                "query": {"multi_match": {"query": query, "fields": ["title^3", "content^2", "citation"]}},
                "size": top_k,
            }
            resp = requests.post(
                f"{ELASTICSEARCH_URL}/deel-legal-chunks/_search",
                json=es_query,
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            sources = []
            for hit in data.get("hits", {}).get("hits", []):
                src = hit["_source"]
                sources.append(UnifiedSource(
                    id=hit["_id"],
                    title=src.get("title", "Unknown"),
                    excerpt=(src.get("content", "") or "")[:500],
                    source_type="bm25",
                    relevance_score=hit.get("_score", 0) / 100.0,  # Normalize
                    metadata=src,
                ))
            return sources
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    async def _search_web(self, query: str, top_k: int = 5) -> List[UnifiedSource]:
        """Search web via SearXNG."""
        try:
            from rag_pipeline.web_search import search_web
            results = await search_web(query, top_k=top_k)
            return [
                UnifiedSource(
                    id=f"web_{hash(r.url) % 10**8}",
                    title=r.title,
                    excerpt=r.snippet[:500],
                    url=r.url,
                    source_type="web",
                    relevance_score=r.score,
                )
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    async def _search_legislation(self, query: str, top_k: int = 5) -> List[UnifiedSource]:
        """Search legislation in PostgreSQL."""
        try:
            from db.database import get_session
            from sqlalchemy import text
            async for db in get_session():
                stmt = text("""
                    SELECT document_id, title, content, source
                    FROM legislation_documents
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    LIMIT :limit
                """)
                result = await db.execute(stmt, {"query": query, "limit": top_k})
                rows = result.fetchall()
                sources = []
                for row in rows:
                    sources.append(UnifiedSource(
                        id=f"leg_{row.document_id}",
                        title=row.title,
                        excerpt=(row.content or "")[:500],
                        source_type="statute",
                        relevance_score=0.7,
                    ))
                return sources
        except Exception as e:
            logger.warning(f"Legislation search failed: {e}")
            return []

    async def _synthesize_answer(
        self, query: str, sources: List[UnifiedSource]
    ) -> tuple[str, List[str]]:
        """Use Gemini to synthesize a cited answer from sources."""
        if not sources:
            return "I couldn't find any relevant sources to answer your question.", []

        # Build context string with source markers
        context_parts = []
        for i, s in enumerate(sources):
            type_label = {"case_law": "Case Law", "web": "Web Source", "statute": "Statute", "bm25": "Case Law"}.get(
                s.source_type, "Source"
            )
            context_parts.append(
                f"[Source {i+1}] ({type_label}) {s.title}\n"
                f"{s.excerpt}\n"
                f"URL: {s.url}\n"
            )
        context = "\n---\n".join(context_parts)

        system_instruction = (
            "You are a senior Canadian legal research assistant. "
            "Synthesize a comprehensive answer from the provided sources. "
            "Cite each claim with the corresponding source number [Source N]. "
            "If sources disagree, note the disagreement. "
            "If no source supports a claim, say 'The available sources do not address this point.'"
        )

        prompt = (
            f"Research question: {query}\n\n"
            f"Available sources:\n{context}\n\n"
            f"Provide a clear, well-structured answer with inline citations. "
            f"Then suggest 2-3 follow-up questions the user might ask next."
        )

        try:
            response = self.chat.generate(
                prompt,
                system_instruction=system_instruction,
                temperature=0.3,
                max_tokens=2048,
            )
            # GeminiChat.generate returns a plain string
            if isinstance(response, str):
                text = response
            else:
                text = response.get("text", "") or response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if not text:
                    text = str(response)

            # Extract follow-up questions (lines after "Follow-up" or last 3 lines starting with "?")
            lines = text.split("\n")
            follow_ups = []
            in_followup = False
            for line in lines:
                stripped = line.strip()
                if "follow-up" in stripped.lower() or "suggested question" in stripped.lower():
                    in_followup = True
                    continue
                if in_followup and stripped and stripped.endswith("?"):
                    follow_ups.append(stripped.lstrip("0123456789. )-"))
                elif in_followup and not stripped:
                    continue
                elif in_followup and stripped:
                    # Non-question line after follow-ups — stop
                    break

            # Clean answer (remove follow-up section)
            answer_lines = []
            for line in lines:
                if "follow-up" in line.lower() and any(c.isdigit() for c in line):
                    break
                answer_lines.append(line)
            answer = "\n".join(answer_lines).strip()

            return answer or text, follow_ups[:5]

        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return (
                f"I found {len(sources)} relevant sources but encountered an error "
                f"generating the analysis. Please try again.",
                [],
            )

    async def deep_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_sources: int = 15,
    ) -> DeepSearchResult:
        """Run a full DeepSearch query across all sources."""
        if not query or len(query.strip()) < 3:
            return DeepSearchResult(
                answer="", error="Query must be at least 3 characters."
            )

        start = time.time()

        # Launch parallel searches
        case_task = self._search_vector(query)
        bm25_task = self._search_bm25(query)
        web_task = self._search_web(query)
        leg_task = self._search_legislation(query)

        case_sources, bm25_sources, web_sources, leg_sources = await asyncio.gather(
            case_task, bm25_task, web_task, leg_task,
            return_exceptions=True,
        )

        # Handle exceptions
        for i, (name, result) in enumerate([
            ("case", case_sources), ("bm25", bm25_sources),
            ("web", web_sources), ("legislation", leg_sources),
        ]):
            if isinstance(result, Exception):
                logger.error(f"{name} search raised: {result}")

        case_sources = case_sources if not isinstance(case_sources, Exception) else []
        bm25_sources = bm25_sources if not isinstance(bm25_sources, Exception) else []
        web_sources = web_sources if not isinstance(web_sources, Exception) else []
        leg_sources = leg_sources if not isinstance(leg_sources, Exception) else []

        # Fuse and rank
        fused = fuse_and_rank_results(
            case_sources + bm25_sources,
            web_sources,
            leg_sources,
            top_k=max_sources,
        )

        # Count by type
        type_counts = count_source_types(fused)

        # Synthesize answer
        answer, follow_ups = await self._synthesize_answer(query, fused)

        elapsed = int((time.time() - start) * 1000)

        return DeepSearchResult(
            answer=answer,
            sources=fused,
            source_type_counts=type_counts,
            suggested_follow_ups=follow_ups,
            processing_time_ms=elapsed,
        )