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
"""

from rag_pipeline.canlii_scraper import CanLIIScraper
from rag_pipeline.document_processor import LegalDocumentProcessor, ProcessedDocument
from rag_pipeline.embeddings import GeminiEmbeddings, GeminiChat
from rag_pipeline.pinecone_client import PineconeClient
from rag_pipeline.rag_query import LegalRAGQuery
from rag_pipeline.pipeline import LegalRAGPipeline

__all__ = [
    "CanLIIScraper",
    "LegalDocumentProcessor",
    "ProcessedDocument",
    "GeminiEmbeddings",
    "GeminiChat",
    "PineconeClient",
    "LegalRAGQuery",
    "LegalRAGPipeline"
]

__version__ = "1.0.0"
