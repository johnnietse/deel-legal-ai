# RAG Pipeline - Main Orchestrator
"""
Main RAG pipeline orchestrator that ties together:
- CanLII web scraping
- Document processing and chunking
- Embedding generation
- Pinecone vector storage
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
from rag_pipeline.document_processor import LegalDocumentProcessor, ProcessedDocument
from rag_pipeline.embeddings import GeminiEmbeddings
from rag_pipeline.pinecone_client import PineconeClient
from rag_pipeline.rag_query import LegalRAGQuery
from config import CANLII_PDF_DOWNLOAD_DIR, DATA_DIR, LOG_FORMAT, LOG_LEVEL

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class LegalRAGPipeline:
    """
    Complete RAG pipeline for legal document ingestion and retrieval.
    
    Workflow:
    1. Scrape documents from CanLII
    2. Extract text and chunk documents
    3. Generate embeddings for each chunk
    4. Upsert to Pinecone vector database
    5. Enable semantic search queries
    """
    
    def __init__(
        self,
        pdf_dir: Path = CANLII_PDF_DOWNLOAD_DIR,
        pinecone_namespace: str = "legal_cases"
    ):
        self.pdf_dir = Path(pdf_dir)
        self.namespace = pinecone_namespace
        
        # Initialize components
        self.scraper = CanLIIScraper(output_dir=self.pdf_dir)
        self.processor = LegalDocumentProcessor(chunk_size=512, chunk_overlap=50)
        self.embeddings = GeminiEmbeddings()
        self.pinecone = PineconeClient()
        self.query_interface = None
        
        # Pipeline state
        self.stats = {
            "documents_scraped": 0,
            "documents_processed": 0,
            "chunks_created": 0,
            "vectors_upserted": 0,
            "errors": []
        }
    
    def run_scraper(
        self, 
        max_cases: Optional[int] = None,
        from_csv: Optional[str] = None
    ) -> int:
        """
        Run the CanLII scraper to download case PDFs.
        
        Args:
            max_cases: Maximum number of cases to scrape (None for all)
            from_csv: Path to CSV file with case URLs
            
        Returns:
            Number of cases successfully scraped
        """
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
        """
        Process all PDFs in directory into chunks.
        
        Args:
            pdf_directory: Directory containing PDFs (uses default if None)
            
        Returns:
            List of ProcessedDocument objects
        """
        pdf_dir = Path(pdf_directory) if pdf_directory else self.pdf_dir
        
        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            return []
        
        logger.info(f"Processing documents from {pdf_dir}...")
        
        documents = self.processor.process_directory(str(pdf_dir))
        
        success_count = sum(1 for d in documents if d.processing_status == "success")
        total_chunks = sum(len(d.chunks) for d in documents)
        
        self.stats["documents_processed"] = success_count
        self.stats["chunks_created"] = total_chunks
        
        logger.info(f"Processed {success_count} documents with {total_chunks} total chunks")
        return documents
    
    def generate_embeddings_for_documents(
        self, 
        documents: List[ProcessedDocument]
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all document chunks.
        
        Args:
            documents: List of processed documents
            
        Returns:
            List of chunks with embeddings
        """
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
        
        # Filter out chunks with failed embeddings
        valid_chunks = [c for c in embedded_chunks if c.get("embedding")]
        
        logger.info(f"Generated {len(valid_chunks)} embeddings")
        return valid_chunks
    
    def upsert_to_pinecone(
        self, 
        embedded_chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Upsert embedded chunks to Pinecone.
        
        Args:
            embedded_chunks: Chunks with embeddings
            
        Returns:
            Number of vectors upserted
        """
        logger.info("Upserting vectors to Pinecone...")
        
        # Create index if needed
        self.pinecone.create_index()
        
        # Upsert documents
        result = self.pinecone.upsert_documents(
            embedded_chunks,
            namespace=self.namespace
        )
        
        upserted_count = result.get("upserted_count", 0)
        self.stats["vectors_upserted"] = upserted_count
        
        logger.info(f"Upserted {upserted_count} vectors")
        return upserted_count
    
    def run_full_pipeline(
        self,
        max_cases: Optional[int] = None,
        skip_scraping: bool = False,
        pdf_directory: Optional[str] = None,
        from_csv: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        
        Args:
            max_cases: Maximum cases to scrape
            skip_scraping: Skip scraping step (use existing PDFs)
            pdf_directory: Custom PDF directory
            from_csv: CSV file to scrape from
            
        Returns:
            Pipeline statistics
        """
        start_time = datetime.now()
        logger.info("="*60)
        logger.info("STARTING LEGAL RAG PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Scrape documents
            if not skip_scraping:
                self.run_scraper(max_cases=max_cases, from_csv=from_csv)
            else:
                logger.info("Skipping scraping step...")
            
            # Step 2: Process documents
            documents = self.process_documents(pdf_directory)
            
            if not documents:
                logger.warning("No documents to process")
                return self.stats
            
            # Step 3: Generate embeddings
            embedded_chunks = self.generate_embeddings_for_documents(documents)
            
            if not embedded_chunks:
                logger.warning("No embeddings generated")
                return self.stats
            
            # Step 4: Upsert to Pinecone
            self.upsert_to_pinecone(embedded_chunks)
            
            # Step 5: Initialize query interface
            self.query_interface = LegalRAGQuery(
                embeddings=self.embeddings,
                pinecone=self.pinecone
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats["errors"].append(str(e))
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        self.stats["duration_seconds"] = duration
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Stats: {json.dumps(self.stats, indent=2)}")
        logger.info("="*60)
        
        return self.stats
    
    def query(self, question: str, **kwargs) -> Any:
        """
        Query the knowledge base.
        
        Args:
            question: Legal question
            **kwargs: Additional query parameters
            
        Returns:
            RAG response
        """
        if self.query_interface is None:
            self.query_interface = LegalRAGQuery(
                embeddings=self.embeddings,
                pinecone=self.pinecone
            )
        
        return self.query_interface.query(question, namespace=self.namespace, **kwargs)


def main():
    """Main entry point for the RAG pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal RAG Pipeline")
    parser.add_argument("--scrape", type=int, default=None, help="Number of cases to scrape")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping step")
    parser.add_argument("--pdf-dir", type=str, default=None, help="PDF directory to process")
    parser.add_argument("--query", type=str, default=None, help="Query the knowledge base")
    parser.add_argument("--csv", type=str, default=None, help="CSV file with cases to scrape")
    
    args = parser.parse_args()
    
    pipeline = LegalRAGPipeline()
    
    if args.query:
        # Query mode
        response = pipeline.query(args.query)
        print(f"\n📝 Query: {response.query}")
        print(f"\n💬 Answer:\n{response.answer}")
        print(f"\n📖 Sources: {len(response.sources)}")
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
