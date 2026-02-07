# RAG Pipeline - Pinecone Vector Database Client
"""
Pinecone integration for legal document vector storage and retrieval.

Implements best practices for legal RAG:
- Serverless index for scalability
- Metadata filtering for jurisdiction, court level, etc.
- Hybrid search support
- Batch upsert for efficiency
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, 
    PINECONE_ENVIRONMENT, PINECONE_DIMENSION, PINECONE_METRIC
)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector similarity search"""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class PineconeClient:
    """
    Pinecone vector database client for legal document retrieval.
    
    Features:
    - Serverless index creation
    - Batch upsert with progress tracking
    - Metadata filtering for legal attributes
    - Similarity search with top-k retrieval
    """
    
    def __init__(
        self,
        api_key: str = PINECONE_API_KEY,
        index_name: str = PINECONE_INDEX_NAME,
        dimension: int = PINECONE_DIMENSION,
        metric: str = PINECONE_METRIC
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
    
    def create_index(self, cloud: str = "aws", region: str = "us-east-1") -> bool:
        """
        Create a serverless Pinecone index if it doesn't exist.
        
        Args:
            cloud: Cloud provider (aws, gcp, azure)
            region: Region for the index
            
        Returns:
            True if index was created, False if it already exists
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name in existing_indexes:
            logger.info(f"Index '{self.index_name}' already exists")
            self.index = self.pc.Index(self.index_name)
            return False
        
        logger.info(f"Creating index '{self.index_name}'...")
        
        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
        
        # Wait for index to be ready
        while not self.pc.describe_index(self.index_name).status['ready']:
            logger.info("Waiting for index to be ready...")
            time.sleep(5)
        
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Index '{self.index_name}' created successfully")
        return True
    
    def connect(self):
        """Connect to existing index"""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        return self.index
    
    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = 100,
        namespace: str = ""
    ) -> Dict[str, int]:
        """
        Upsert vectors to Pinecone index.
        
        Args:
            vectors: List of vectors with format:
                     {"id": str, "values": List[float], "metadata": Dict}
            batch_size: Number of vectors per batch
            namespace: Optional namespace for organization
            
        Returns:
            Stats about the upsert operation
        """
        if self.index is None:
            self.connect()
        
        from tqdm import tqdm
        
        total_upserted = 0
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting vectors"):
            batch = vectors[i:i + batch_size]
            
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
            except Exception as e:
                logger.error(f"Error upserting batch {i}: {e}")
        
        return {"upserted_count": total_upserted}
    
    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        id_key: str = "chunk_id",
        embedding_key: str = "embedding",
        content_key: str = "content",
        namespace: str = ""
    ) -> Dict[str, int]:
        """
        Upsert documents with their embeddings.
        
        Args:
            documents: List of documents with embeddings
            id_key: Key for document ID
            embedding_key: Key for embedding vector
            content_key: Key for text content
            namespace: Optional namespace
            
        Returns:
            Stats about the upsert operation
        """
        vectors = []
        
        for doc in documents:
            embedding = doc.get(embedding_key, [])
            if not embedding:
                logger.warning(f"Document {doc.get(id_key)} has no embedding, skipping")
                continue
            
            # Prepare metadata (Pinecone has limits on metadata size)
            metadata = {
                "content": doc.get(content_key, "")[:10000],  # Limit content size
                "document_id": doc.get("document_id", ""),
                "chunk_index": doc.get("chunk_index", 0),
            }
            
            # Add any additional metadata from the document
            if "metadata" in doc:
                for key, value in doc["metadata"].items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, list):
                        metadata[key] = value[:10]  # Limit list size
            
            vectors.append({
                "id": doc.get(id_key),
                "values": embedding,
                "metadata": metadata
            })
        
        return self.upsert_vectors(vectors, namespace=namespace)
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            namespace: Namespace to search in
            filter: Metadata filter (e.g., {"jurisdiction": "ON"})
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            self.connect()
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=include_metadata
            )
            
            search_results = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                search_results.append(SearchResult(
                    id=match["id"],
                    score=match["score"],
                    content=metadata.get("content", ""),
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    def delete_namespace(self, namespace: str):
        """Delete all vectors in a namespace"""
        if self.index is None:
            self.connect()
        
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted all vectors in namespace '{namespace}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if self.index is None:
            self.connect()
        
        return self.index.describe_index_stats()


def test_pinecone():
    """Test Pinecone client"""
    client = PineconeClient()
    
    print("\n" + "="*60)
    print("TESTING PINECONE CLIENT")
    print("="*60)
    
    try:
        # Create or connect to index
        created = client.create_index()
        print(f"\n{'Created' if created else 'Connected to'} index: {client.index_name}")
        
        # Get stats
        stats = client.get_stats()
        print(f"Index stats: {stats}")
        
        # Test upsert with dummy data
        test_vectors = [
            {
                "id": "test_1",
                "values": [0.1] * PINECONE_DIMENSION,
                "metadata": {"content": "Test legal document 1", "jurisdiction": "ON"}
            },
            {
                "id": "test_2", 
                "values": [0.2] * PINECONE_DIMENSION,
                "metadata": {"content": "Test legal document 2", "jurisdiction": "BC"}
            }
        ]
        
        result = client.upsert_vectors(test_vectors, namespace="test")
        print(f"\n✅ Upserted {result['upserted_count']} vectors")
        
        # Test search
        results = client.search(
            query_vector=[0.15] * PINECONE_DIMENSION,
            top_k=2,
            namespace="test"
        )
        print(f"\n✅ Search returned {len(results)} results")
        for r in results:
            print(f"   - {r.id}: score={r.score:.3f}")
        
        # Cleanup test data
        client.delete_namespace("test")
        print("\n✅ Cleaned up test namespace")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pinecone()
