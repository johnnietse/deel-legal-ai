# RAG Pipeline - Main Orchestrator
"""
Main RAG pipeline orchestrator that ties together:
- CanLII web scraping
- Document processing and semantic chunking (v3.0)
- Embedding generation
- Vector storage (Pinecone / Milvus) (v3.0)
- BM25 indexing for Hybrid Search (v3.0)
- Query interface

This implements the full pipeline described in the resume:
"Engineered a scalable legal data ingestion RAG pipeline using Python 
(BeautifulSoup, Selenium) to automate the collection and preprocessing 
of 10,000+ case law documents from CanLII, then chunking and embedding 
legal statutes into Pinecone vector database"
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.canlii_scraper import CanLIIScraper
from rag_pipeline.document_processor import LegalDocumentProcessor, ProcessedDocument, SemanticChunker
from rag_pipeline.embeddings import GeminiEmbeddings
from rag_pipeline.vector_store import create_vector_store
from rag_pipeline.search_engine import create_bm25_engine
from rag_pipeline.rag_query import LegalRAGQuery
from config import CANLII_PDF_DOWNLOAD_DIR, DATA_DIR, LOG_FORMAT, LOG_LEVEL

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class LegalRAGPipeline:
    """
    Complete RAG pipeline for legal document ingestion and retrieval.
    
    Workflow (v3.0):
    1. Scrape documents from CanLII
    2. Extract text and semantically chunk documents (preserving legal sections)
    3. Generate embeddings for each chunk
    4. Upsert to Vector Store (Pinecone/Milvus)
    5. Index into Sparse Store (Elasticsearch/Local BM25)
    6. Enable hybrid semantic search queries
    """
    
    def __init__(
        self,
        pdf_dir: Path = CANLII_PDF_DOWNLOAD_DIR,
        namespace: str = "legal_cases"
    ):
        self.pdf_dir = Path(pdf_dir)
        self.namespace = namespace
        
        # Initialize components (v3.0)
        self.scraper = CanLIIScraper(output_dir=self.pdf_dir)
        self.processor = LegalDocumentProcessor(chunk_size=512, chunk_overlap=50)
        self.semantic_chunker = SemanticChunker()
        self.embeddings = GeminiEmbeddings()
        
        # Dual-store indexing (v3.0)
        self.vector_store = create_vector_store()
        self.bm25_engine = create_bm25_engine()
        self.query_interface = None
        
        # Pipeline state
        self.stats = {
            "documents_scraped": 0,
            "documents_processed": 0,
            "chunks_created": 0,
            "vectors_upserted": 0,
            "bm25_indexed": 0,
            "errors": []
        }
    
    def run_scraper(
        self, 
        max_cases: Optional[int] = None,
        from_csv: Optional[str] = None
    ) -> int:
        """Run CanLII scraper to download case PDFs."""
        logger.info("Starting CanLII scraper...")
        
        if from_csv:
            results = self.scraper.scrape_from_csv(from_csv, max_cases=max_cases)
        else:
            results = self.scraper.scrape_employment_cases(max_cases=max_cases)
        
        success_count = sum(1 for r in results if r.status == "success")
        self.stats["documents_scraped"] = success_count
        
        logger.info(f"Scraping complete: {success_count}/{len(results)} successful")
        return success_count
    
    def process_documents(self, pdf_directory: Optional[str] = None) -> List[ProcessedDocument]:
        """Process PDFs into semantic chunks (v3.0 single-pass)."""
        pdf_dir = Path(pdf_directory) if pdf_directory else self.pdf_dir

        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            return []

        logger.info(f"Processing documents from {pdf_dir} using SemanticChunker...")

        # Extract full text from PDFs, then semantically chunk directly
        documents = self.processor.process_directory(str(pdf_dir))

        success_count = 0
        total_chunks = 0

        for doc in documents:
            if doc.processing_status == "success":
                semantic_chunks = self.semantic_chunker.chunk_document(
                    doc.full_text,
                    doc_id=doc.document_id,
                    metadata=doc.metadata
                )

                doc.chunks = semantic_chunks
                success_count += 1
                total_chunks += len(semantic_chunks)

        self.stats["documents_processed"] = success_count
        self.stats["chunks_created"] = total_chunks

        logger.info(f"Semantically processed {success_count} documents with {total_chunks} total chunks")
        return documents
    
    def generate_embeddings_for_documents(
        self, 
        documents: List[ProcessedDocument]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for all document chunks."""
        logger.info("Generating embeddings for document chunks...")
        
        all_chunks = []
        for doc in documents:
            if doc.processing_status != "success":
                continue
            
            for chunk in doc.chunks:
                all_chunks.append(chunk.to_dict())
        
        if not all_chunks:
            logger.warning("No chunks to embed")
            return []
        
        # Generate embeddings in batches
        embedded_chunks = self.embeddings.embed_documents(all_chunks)
        
        valid_chunks = [c for c in embedded_chunks if c.get("embedding")]
        logger.info(f"Generated {len(valid_chunks)} embeddings")
        return valid_chunks
    
    def upsert_to_stores(
        self, 
        embedded_chunks: List[Dict[str, Any]]
    ) -> None:
        """Upsert chunks to both Vector Store and BM25 Sparse Store."""
        logger.info("Upserting vectors to Vector Store...")
        
        # Vector Store (Pinecone/Milvus)
        try:
            self.vector_store.create_index()
            upserted_count = self.vector_store.upsert_documents(
                embedded_chunks,
                namespace=self.namespace
            )
            self.stats["vectors_upserted"] = upserted_count
            logger.info(f"Upserted {upserted_count} vectors to {self.vector_store.__class__.__name__}")
        except Exception as e:
            logger.error(f"Vector Store upsert failed: {e}")
            self.stats["errors"].append(f"VectorStore: {e}")
            
        # Sparse Store (BM25 Elasticsearch/Local)
        logger.info("Indexing to BM25 Sparse Store...")
        try:
            # We don't need the dense embeddings for BM25, just the text/metadata
            self.bm25_engine.create_index()
            indexed_count = self.bm25_engine.index_documents(
                embedded_chunks,
                namespace=self.namespace
            )
            self.stats["bm25_indexed"] = indexed_count
            logger.info(f"Indexed {indexed_count} documents to {self.bm25_engine.__class__.__name__}")
        except Exception as e:
            logger.error(f"BM25 Store index failed: {e}")
            self.stats["errors"].append(f"BM25Store: {e}")
            
    def run_full_pipeline(
        self,
        max_cases: Optional[int] = None,
        skip_scraping: bool = False,
        pdf_directory: Optional[str] = None,
        from_csv: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete RAG ingestion pipeline."""
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("STARTING LEGAL RAG PIPELINE (v3.0)")
        logger.info("="*60)
        
        try:
            # Step 1: Scrape documents
            if not skip_scraping:
                self.run_scraper(max_cases=max_cases, from_csv=from_csv)
            else:
                logger.info("Skipping scraping step...")
            
            # Step 2: Process documents (Semantic Chunking)
            documents = self.process_documents(pdf_directory)
            if not documents:
                logger.warning("No documents to process")
                return self.stats
            
            # Step 3: Generate embeddings
            embedded_chunks = self.generate_embeddings_for_documents(documents)
            if not embedded_chunks:
                logger.warning("No embeddings generated")
                return self.stats
            
            # Step 4 & 5: Dual Upsert
            self.upsert_to_stores(embedded_chunks)
            
            # Step 6: Initialize query interface
            self.query_interface = LegalRAGQuery()
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats["errors"].append(str(e))
        
        duration = (datetime.now() - start_time).total_seconds()
        self.stats["duration_seconds"] = duration
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Stats: {json.dumps(self.stats, indent=2)}")
        logger.info("="*60)
        
        return self.stats
    
    def query(self, question: str, **kwargs) -> Any:
        """Query the knowledge base using the v3.0 pipeline."""
        if self.query_interface is None:
            self.query_interface = LegalRAGQuery()
        
        return self.query_interface.query(question, namespace=self.namespace, **kwargs)


def main():
    """Main entry point for the RAG pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal RAG Pipeline (v3.0)")
    parser.add_argument("--scrape", type=int, default=None, help="Number of cases to scrape")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping step")
    parser.add_argument("--pdf-dir", type=str, default=None, help="PDF directory to process")
    parser.add_argument("--query", type=str, default=None, help="Query the knowledge base")
    parser.add_argument("--csv", type=str, default=None, help="CSV file with cases to scrape")
    parser.add_argument("--reindex", action="store_true", help="Reindex existing local JSON/PDF files to both Vector and Sparse stores")
    
    args = parser.parse_args()
    
    pipeline = LegalRAGPipeline()
    
    if args.query:
        # Query mode
        result = pipeline.query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"\nAnswer:\n{result.get('answer', 'No answer generated')}")
        print(f"\nSources: {len(result.get('sources', []))}")
        if "metrics" in result:
            print(f"\nLatency: {result['metrics'].get('total_latency_ms', 0)}ms")
    elif args.reindex:
        logger.info("=== REINDEX MODE ===")
        logger.info("Re-indexing existing documents to both Vector Store and BM25 Sparse Store")
        pipeline.run_full_pipeline(
            max_cases=None,
            skip_scraping=True,
            pdf_directory=args.pdf_dir,
            from_csv=None
        )
    else:
        # Pipeline mode
        pipeline.run_full_pipeline(
            max_cases=args.scrape,
            skip_scraping=args.skip_scrape,
            pdf_directory=args.pdf_dir,
            from_csv=args.csv
        )


if __name__ == "__main__":
    main()

