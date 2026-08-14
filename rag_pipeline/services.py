"""RAG service factories — single source of truth for pipeline construction.

All user-facing entry points (rag_query endpoint, SSE streaming, DeepSearch)
construct their LegalRAGQuery through these factories so they share the same
search_key_manager pool, cache, and instance semantics instead of building
per-request pipelines with default (embedder) keys.
"""
import threading
import logging

logger = logging.getLogger(__name__)

_build_lock = threading.Lock()


def get_search_key_manager():
    """Return the dedicated user-facing key pool, falling back to the shared
    embedder pool if the search pool is unconfigured."""
    from rag_pipeline.gemini_key_manager import key_manager, search_key_manager
    if search_key_manager.key_count == 0:
        logger.warning("search_key_manager has 0 keys — falling back to shared embedder pool")
        return key_manager
    return search_key_manager


def build_search_services():
    """Build (embeddings, chat) wired to the dedicated search key pool.

    Chat uses MultiModelChat: Groq primary, Gemini fallback. A single
    provider outage never degrades the user-facing answer.
    """
    from rag_pipeline.embeddings import GeminiEmbeddings, MultiModelChat
    km = get_search_key_manager()
    return GeminiEmbeddings(key_manager=km), MultiModelChat()


def build_rag_query():
    """Build a LegalRAGQuery wired to the search key pool (thread-safe)."""
    from config import PINECONE_INDEX_NAME, VECTOR_STORE_BACKEND
    from rag_pipeline.vector_store import create_vector_store
    from rag_pipeline.rag_query import LegalRAGQuery

    with _build_lock:
        embeddings, chat = build_search_services()
        return LegalRAGQuery(
            vector_store=create_vector_store(
                backend=VECTOR_STORE_BACKEND,
                index_name=PINECONE_INDEX_NAME,
            ),
            embeddings=embeddings,
            chat=chat,
        )
