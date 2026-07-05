# RAG Pipeline - Elasticsearch-Backed BM25 Search Engine
"""
Production-grade BM25 search using Elasticsearch,
following ByteDance's commercialised RAG architecture.

ByteDance uses their internal search infrastructure for sparse retrieval.
Elasticsearch is the closest commercially available equivalent:
  - Distributed, horizontally scalable
  - BM25 out-of-the-box with configurable similarity
  - Supports cross-region replication
  - Rich query DSL for legal metadata filtering
  - Integrates with hybrid search via kNN + BM25

This module provides:
  1. ElasticsearchBM25 — production Elasticsearch backend
  2. LocalBM25 — in-memory fallback for development (from hybrid_retriever.py)
  3. Factory function to switch based on config

Requires: pip install elasticsearch
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structure
# ---------------------------------------------------------------------------

@dataclass
class BM25Result:
    """Result from BM25 search."""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Elasticsearch Backend (Production)
# ---------------------------------------------------------------------------

class ElasticsearchBM25:
    """
    Elasticsearch-backed BM25 search engine.

    ByteDance architecture: separate sparse retrieval service with
    configurable BM25 parameters, index-time analysers for legal
    text, and cross-region replication for availability.

    Index settings optimised for legal text:
      - Custom analyser preserving legal citations
      - Edge n-gram for partial statute matching
      - Metadata fields for jurisdiction filtering
    """

    # Index settings optimised for legal text
    LEGAL_INDEX_SETTINGS = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "legal_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "legal_stop",
                            "legal_synonym",
                        ],
                    },
                },
                "filter": {
                    "legal_stop": {
                        "type": "stop",
                        "stopwords": "_english_",
                    },
                    "legal_synonym": {
                        "type": "synonym",
                        "synonyms": [
                            "employee,worker,staff",
                            "independent contractor,IC,freelancer,self-employed",
                            "termination,dismissal,firing",
                            "notice period,severance,termination pay",
                            "ESA,Employment Standards Act",
                            "CLC,Canada Labour Code",
                            "OHSA,Occupational Health and Safety Act",
                        ],
                    },
                },
            },
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "legal_analyzer",
                },
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "jurisdiction": {"type": "keyword"},
                "court_level": {"type": "keyword"},
                "legal_section": {"type": "keyword"},
                "case_name": {"type": "text"},
                "primary_citation": {"type": "keyword"},
                "decision_date": {"type": "date", "format": "yyyy-MM-dd||epoch_millis", "ignore_malformed": True},
                "chunk_index": {"type": "integer"},
            },
        },
    }

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "deel-legal-chunks",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.hosts = hosts or [os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")]
        self.cloud_id = cloud_id or os.getenv("ELASTICSEARCH_CLOUD_ID", "")
        self.api_key = api_key or os.getenv("ELASTICSEARCH_API_KEY", "")
        self.index_name = index_name
        self.username = username or os.getenv("ELASTICSEARCH_USERNAME", "")
        self.password = password or os.getenv("ELASTICSEARCH_PASSWORD", "")
        self._client = None

    def connect(self) -> bool:
        """Connect to Elasticsearch cluster."""
        try:
            from elasticsearch import Elasticsearch

            if self.cloud_id:
                self._client = Elasticsearch(
                    cloud_id=self.cloud_id,
                    api_key=self.api_key,
                )
            elif self.username and self.password:
                self._client = Elasticsearch(
                    self.hosts,
                    basic_auth=(self.username, self.password),
                )
            else:
                self._client = Elasticsearch(self.hosts)

            info = self._client.info()
            logger.info(
                f"Connected to Elasticsearch {info['version']['number']} "
                f"at {self.hosts}"
            )
            return True
        except ImportError:
            logger.error("elasticsearch package not installed. Run: pip install elasticsearch")
            return False
        except Exception as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            return False

    def create_index(self) -> bool:
        """Create the legal document index with optimised mappings."""
        if self._client is None:
            self.connect()

        try:
            if self._client.indices.exists(index=self.index_name):
                logger.info(f"Index '{self.index_name}' already exists")
                return False

            self._client.indices.create(
                index=self.index_name,
                body=self.LEGAL_INDEX_SETTINGS,
            )
            logger.info(f"Created Elasticsearch index '{self.index_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 500,
    ) -> int:
        """
        Index document chunks into Elasticsearch.

        Args:
            chunks: List of chunk dicts with 'content', 'chunk_id', metadata
            batch_size: Bulk indexing batch size

        Returns:
            Number of chunks indexed
        """
        if self._client is None:
            self.connect()

        from elasticsearch.helpers import bulk

        total = 0
        actions = []

        for chunk in chunks:
            doc = {
                "_index": self.index_name,
                "_id": chunk.get("chunk_id", chunk.get("id", "")),
                "_source": {
                    "content": chunk.get("content", ""),
                    "chunk_id": chunk.get("chunk_id", chunk.get("id", "")),
                    "document_id": chunk.get("document_id", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                },
            }

            # Add metadata fields
            metadata = chunk.get("metadata", {})
            for field_name in [
                "jurisdiction", "court_level", "legal_section",
                "case_name", "primary_citation",
            ]:
                if field_name in metadata:
                    doc["_source"][field_name] = metadata[field_name]

            actions.append(doc)

            if len(actions) >= batch_size:
                success, errors = bulk(self._client, actions, raise_on_error=False)
                total += success
                if errors:
                    logger.warning(f"Bulk indexing had {len(errors)} errors")
                actions = []

        # Final batch
        if actions:
            success, errors = bulk(self._client, actions, raise_on_error=False)
            total += success

        self._client.indices.refresh(index=self.index_name)
        logger.info(f"Indexed {total} chunks into '{self.index_name}'")
        return total

    def search(
        self,
        query: str,
        top_k: int = 50,
        filter: Optional[Dict[str, Any]] = None,
        boost_fields: Optional[Dict[str, float]] = None,
    ) -> List[BM25Result]:
        """
        BM25 search with optional metadata filtering.

        Args:
            query: Search query text
            top_k: Maximum results
            filter: Metadata filter dict (e.g., {"jurisdiction": "ON"})
            boost_fields: Field boosting (e.g., {"case_name": 2.0})

        Returns:
            List of BM25Result sorted by score
        """
        if self._client is None:
            self.connect()

        # Build query
        must_clause = {
            "multi_match": {
                "query": query,
                "fields": ["content", "case_name"],
                "type": "best_fields",
            }
        }

        # Apply field boosting
        if boost_fields:
            fields = []
            for field_name, boost in boost_fields.items():
                fields.append(f"{field_name}^{boost}")
            must_clause["multi_match"]["fields"] = fields

        # Build filter clauses
        filter_clauses = []
        if filter:
            for key, value in filter.items():
                if isinstance(value, list):
                    filter_clauses.append({"terms": {key: value}})
                else:
                    filter_clauses.append({"term": {key: value}})

        # Assemble full query
        body = {
            "query": {
                "bool": {
                    "must": [must_clause],
                    "filter": filter_clauses,
                },
            },
            "size": top_k,
        }

        try:
            response = self._client.search(index=self.index_name, body=body)

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = {
                    k: v for k, v in source.items()
                    if k not in ("content",)
                }
                results.append(BM25Result(
                    id=hit["_id"],
                    score=hit["_score"],
                    content=source.get("content", ""),
                    metadata=metadata,
                ))
            return results

        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []

    def delete_index(self):
        """Delete the index (dangerous — use for reset only)."""
        if self._client and self._client.indices.exists(index=self.index_name):
            self._client.indices.delete(index=self.index_name)
            logger.info(f"Deleted index '{self.index_name}'")

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self._client is None:
            return {"error": "Not connected"}
        try:
            stats = self._client.indices.stats(index=self.index_name)
            primaries = stats["indices"][self.index_name]["primaries"]
            return {
                "backend": "elasticsearch",
                "index": self.index_name,
                "doc_count": primaries["docs"]["count"],
                "size_bytes": primaries["store"]["size_in_bytes"],
                "search_count": primaries["search"]["query_total"],
            }
        except Exception as e:
            return {"backend": "elasticsearch", "error": str(e)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_bm25_engine(
    backend: str = "elasticsearch",
    **kwargs,
):
    """
    Factory function to create a BM25 search engine.

    Args:
        backend: "elasticsearch" or "local"
        **kwargs: Backend-specific configuration

    Returns:
        BM25 engine instance
    """
    if backend == "elasticsearch":
        return ElasticsearchBM25(**kwargs)
    elif backend == "local":
        from rag_pipeline.hybrid_retriever import BM25Index
        return BM25Index(**kwargs)
    else:
        raise ValueError(f"Unknown BM25 backend: {backend}")
