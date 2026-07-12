#!/usr/bin/env python3
"""
Migrate vectors from Pinecone to local Milvus.
Run this once to populate local Milvus with all vectors from Pinecone.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinecone import Pinecone
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PINECONE_INDEX_NAME = PINECONE_INDEX_NAME or "deel-legal-cases"
MILVUS_COLLECTION_NAME = "deel_legal_cases"
MILVUS_DIMENSION = 3072  # gemini-embedding-001 dimension
BATCH_SIZE = 100

def connect_pinecone():
    """Connect to Pinecone and return index."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    return index

def connect_milvus():
    """Connect to local Milvus."""
    connections.connect("default", host="localhost", port="19530")
    logger.info("Connected to local Milvus")

def create_milvus_collection():
    """Create Milvus collection with same schema as Pinecone."""
    if utility.has_collection(MILVUS_COLLECTION_NAME):
        logger.info(f"Collection {MILVUS_COLLECTION_NAME} already exists, dropping...")
        utility.drop_collection(MILVUS_COLLECTION_NAME)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_DIMENSION),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="case_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="year", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="jurisdiction", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="citations", dtype=DataType.VARCHAR, max_length=500),
    ]
    
    schema = CollectionSchema(fields, description="Legal cases from Pinecone migration")
    collection = Collection(MILVUS_COLLECTION_NAME, schema)
    logger.info(f"Created Milvus collection: {MILVUS_COLLECTION_NAME}")
    return collection

def create_milvus_index(collection):
    """Create HNSW index on vector field."""
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    logger.info("Created HNSW index on vector field")

def get_all_pinecone_ids(pinecone_index):
    """Get all vector IDs from Pinecone legal_cases namespace."""
    all_ids = []
    for batch in pinecone_index.list(namespace='legal_cases', limit=100):
        for item in batch:
            all_ids.append(item.id)
    logger.info(f"Found {len(all_ids)} vectors in Pinecone legal_cases namespace")
    return all_ids

def migrate_batch(pinecone_index, milvus_collection, batch_ids):
    """Migrate a batch of vectors from Pinecone to Milvus."""
    # Fetch from Pinecone
    fetch_response = pinecone_index.fetch(ids=batch_ids, namespace='legal_cases')
    
    if not fetch_response.vectors:
        logger.warning(f"No vectors found for batch: {batch_ids}")
        return 0
    
    # Prepare data for Milvus
    ids = []
    vectors = []
    titles = []
    contents = []
    sources = []
    case_types = []
    years = []
    jurisdictions = []
    topics = []
    citations = []
    
    for vec_id, vec_data in fetch_response.vectors.items():
        metadata = vec_data.metadata or {}
        ids.append(vec_id)
        vectors.append(vec_data.values)
        titles.append(metadata.get("title", "")[:511])
        contents.append(metadata.get("content", "")[:65534])
        sources.append(metadata.get("source", "")[:99])
        case_types.append(metadata.get("case_type", "")[:99])
        years.append(metadata.get("year", "")[:9])
        jurisdictions.append(metadata.get("jurisdiction", "")[:99])
        topics.append(metadata.get("topic", "")[:199])
        citations.append(metadata.get("citations", "")[:499])
    
    # Insert into Milvus
    entities = [
        ids, vectors, titles, contents, sources,
        case_types, years, jurisdictions, topics, citations
    ]
    
    milvus_collection.insert(entities)
    milvus_collection.flush()
    
    logger.info(f"Migrated {len(ids)} vectors to Milvus")
    return len(ids)

def main():
    logger.info("Starting Pinecone to Milvus migration...")
    
    # Connect to both databases
    pinecone_index = connect_pinecone()
    connect_milvus()
    
    # Create Milvus collection
    collection = create_milvus_collection()
    create_milvus_index(collection)
    
    # Get all vector IDs from Pinecone
    all_ids = get_all_pinecone_ids(pinecone_index)
    logger.info(f"Total vectors to migrate: {len(all_ids)}")
    
    # Migrate in batches
    total_migrated = 0
    failed = 0
    
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        logger.info(f"Migrating batch {i//BATCH_SIZE + 1}/{(len(all_ids) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_ids)} vectors)")
        
        try:
            migrated = migrate_batch(pinecone_index, collection, batch_ids)
            total_migrated += migrated
            logger.info(f"Progress: {total_migrated}/{len(all_ids)} vectors migrated")
        except Exception as e:
            logger.error(f"Failed to migrate batch: {e}")
            failed += len(batch_ids)
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    logger.info(f"Migration complete! Total migrated: {total_migrated}, Failed: {failed}")
    
    # Verify
    collection.flush()
    stats = collection.num_entities
    logger.info(f"Final Milvus collection count: {stats}")

if __name__ == "__main__":
    main()