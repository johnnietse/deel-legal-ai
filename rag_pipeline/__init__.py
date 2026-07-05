# RAG Pipeline Package
"""
Legal RAG Pipeline for Deel Lab / OpenJustice.ai

Components:
- canlii_scraper: Web scraping from CanLII
- document_processor: PDF extraction and chunking (+ SemanticChunker)
- embeddings: Gemini API embeddings
- pinecone_client: Vector database operations
- rag_query: Query interface (ByteDance-enhanced)
- pipeline: Main orchestrator
- multi_hop_retriever: Multi-hop reasoning retrieval
- knowledge_graph: Legal knowledge graph construction and traversal
- graph_retriever: Hybrid vector + graph retrieval
- legal_reasoning_agent: MCTS-based legal reasoning

ByteDance RAG Enhancements (§4-§8):
- hybrid_retriever: BM25 + vector fusion with RRF and MMR
- prompt_templates: Domain-specific prompt templates with auto-selection
- confidence_gate: Pre-generation confidence checking
- query_cache: Multi-layer TTL cache (embedding, retrieval, response)
- metrics: Full-pipeline instrumentation and structured logging
- feedback_analyzer: User feedback collection and analysis
- vector_store: Abstract vector store (Pinecone / Milvus)
- search_engine: Elasticsearch-backed BM25
- model_optimization: LoRA, quantisation, distillation scaffolding
"""

# ---------------------------------------------------------------------------
# Lazy imports — only resolve when accessed.
#
# The package has modules with heavy third-party dependencies (bs4, fitz,
# selenium, pinecone, etc.).  Eager imports in __init__.py cause ImportError
# if any optional dependency is missing.  We keep the public API identical
# (users can still `from rag_pipeline import LegalRAGQuery`) by using
# module-level __getattr__ for lazy loading.
# ---------------------------------------------------------------------------

import importlib as _importlib

# Map of public name -> (module_path, attribute_name)
_LAZY_IMPORTS = {
    # Core pipeline
    "CanLIIScraper":          ("rag_pipeline.canlii_scraper", "CanLIIScraper"),
    "LegalDocumentProcessor": ("rag_pipeline.document_processor", "LegalDocumentProcessor"),
    "ProcessedDocument":      ("rag_pipeline.document_processor", "ProcessedDocument"),
    "SemanticChunker":        ("rag_pipeline.document_processor", "SemanticChunker"),
    "GeminiEmbeddings":       ("rag_pipeline.embeddings", "GeminiEmbeddings"),
    "GeminiChat":             ("rag_pipeline.embeddings", "GeminiChat"),
    "PineconeClient":         ("rag_pipeline.pinecone_client", "PineconeClient"),
    "LegalRAGQuery":          ("rag_pipeline.rag_query", "LegalRAGQuery"),
    "LegalRAGPipeline":       ("rag_pipeline.pipeline", "LegalRAGPipeline"),
    "MultiHopRetriever":      ("rag_pipeline.multi_hop_retriever", "MultiHopRetriever"),
    "MultiHopResult":         ("rag_pipeline.multi_hop_retriever", "MultiHopResult"),
    "LegalKnowledgeGraph":    ("rag_pipeline.knowledge_graph", "LegalKnowledgeGraph"),
    "GraphHybridRetriever":   ("rag_pipeline.graph_retriever", "HybridRetriever"),
    "LegalReasoningAgent":    ("rag_pipeline.legal_reasoning_agent", "LegalReasoningAgent"),
    # ByteDance enhancements
    "HybridRetriever":        ("rag_pipeline.hybrid_retriever", "HybridRetriever"),
    "HybridResult":           ("rag_pipeline.hybrid_retriever", "HybridResult"),
    "BM25Index":              ("rag_pipeline.hybrid_retriever", "BM25Index"),
    "QueryClassifier":        ("rag_pipeline.hybrid_retriever", "QueryClassifier"),
    "PromptTemplateLibrary":  ("rag_pipeline.prompt_templates", "PromptTemplateLibrary"),
    "ConfidenceGate":         ("rag_pipeline.confidence_gate", "ConfidenceGate"),
    "RAGQueryCache":          ("rag_pipeline.query_cache", "RAGQueryCache"),
    "MetricsCollector":       ("rag_pipeline.metrics", "MetricsCollector"),
    "QueryMetrics":           ("rag_pipeline.metrics", "QueryMetrics"),
    "FeedbackStore":          ("rag_pipeline.feedback_analyzer", "FeedbackStore"),
    "FeedbackAnalyzer":       ("rag_pipeline.feedback_analyzer", "FeedbackAnalyzer"),
    "VectorStore":            ("rag_pipeline.vector_store", "VectorStore"),
    "PineconeBackend":        ("rag_pipeline.vector_store", "PineconeBackend"),
    "MilvusBackend":          ("rag_pipeline.vector_store", "MilvusBackend"),
    "create_vector_store":    ("rag_pipeline.vector_store", "create_vector_store"),
    "ElasticsearchBM25":      ("rag_pipeline.search_engine", "ElasticsearchBM25"),
    "create_bm25_engine":     ("rag_pipeline.search_engine", "create_bm25_engine"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'rag_pipeline' has no attribute {name!r}")


__all__ = list(_LAZY_IMPORTS.keys())

__version__ = "3.0.0"
