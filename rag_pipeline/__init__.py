# RAG Pipeline Package
"""
Legal RAG Pipeline for Deel Lab / OpenJustice.ai

Components:
- canlii_scraper: Web scraping from CanLII
- document_processor: PDF extraction and chunking
- embeddings: Gemini API embeddings
- pinecone_client: Vector database operations
- rag_query: Query interface
- pipeline: Main orchestrator
- multi_hop_retriever: Multi-hop reasoning retrieval
- knowledge_graph: Legal knowledge graph construction and traversal
- graph_retriever: Hybrid vector + graph retrieval
- legal_reasoning_agent: MCTS-based legal reasoning
"""

from rag_pipeline.canlii_scraper import CanLIIScraper
from rag_pipeline.document_processor import LegalDocumentProcessor, ProcessedDocument
from rag_pipeline.embeddings import GeminiEmbeddings, GeminiChat
from rag_pipeline.pinecone_client import PineconeClient
from rag_pipeline.rag_query import LegalRAGQuery
from rag_pipeline.pipeline import LegalRAGPipeline
from rag_pipeline.multi_hop_retriever import MultiHopRetriever, MultiHopResult
from rag_pipeline.knowledge_graph import LegalKnowledgeGraph
from rag_pipeline.graph_retriever import HybridRetriever
from rag_pipeline.legal_reasoning_agent import LegalReasoningAgent

__all__ = [
    "CanLIIScraper",
    "LegalDocumentProcessor",
    "ProcessedDocument",
    "GeminiEmbeddings",
    "GeminiChat",
    "PineconeClient",
    "LegalRAGQuery",
    "LegalRAGPipeline",
    "MultiHopRetriever",
    "MultiHopResult",
    "LegalKnowledgeGraph",
    "HybridRetriever",
    "LegalReasoningAgent",
]

__version__ = "2.0.0"
