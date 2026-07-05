#!/usr/bin/env python3
"""
Elasticsearch Bootstrap Script (Deployment Phase Item 2)

Connects to a running Elasticsearch instance (from docker-compose),
creates the legal index with the custom analyzer, and bulk-indexes
all semantic chunks from the local re-index output.

Usage:
  1. Start Elasticsearch: docker-compose up elasticsearch -d
  2. Run this script:     python scripts/bootstrap_elasticsearch.py

Prerequisites:
  pip install elasticsearch
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("  DEEL LEGAL AI -- Elasticsearch Bootstrap")
    print("=" * 70)
    print()

    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    print(f"[1/4] Connecting to Elasticsearch at {es_url}...")

    from rag_pipeline.search_engine import ElasticsearchBM25

    engine = ElasticsearchBM25(hosts=[es_url])
    connected = engine.connect()

    if not connected:
        print("FAILED: Could not connect to Elasticsearch.")
        print("Make sure Elasticsearch is running:")
        print("  docker-compose up elasticsearch -d")
        return False

    print("  Connected!")
    print()

    # Step 2: Create index
    print("[2/4] Creating legal index with custom analyzer...")
    created = engine.create_index()
    if created:
        print(f"  Created index '{engine.index_name}'")
    else:
        print(f"  Index '{engine.index_name}' already exists (skipping)")
    print()

    # Step 3: Load chunks from the local re-index output
    chunks_path = Path("data/semantic_index/chunks.jsonl")
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run reindex_local.py first.")
        return False

    print(f"[3/4] Loading chunks from {chunks_path}...")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"  Loaded {len(chunks)} chunks")
    print()

    # Step 4: Bulk index into Elasticsearch
    print(f"[4/4] Bulk indexing {len(chunks)} chunks...")
    start = time.time()
    indexed = engine.index_chunks(chunks)
    elapsed = time.time() - start

    print(f"  Indexed {indexed} chunks in {elapsed:.1f}s")
    print()

    # Verify
    stats = engine.stats()
    print("Index Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    # Run a test query
    print("Test Query: 'Sagaz test worker classification'")
    results = engine.search("Sagaz test worker classification", top_k=3)
    for i, r in enumerate(results):
        snippet = r.content[:80].replace("\n", " ").strip()
        print(f"  [{i+1}] {r.id} (score: {r.score:.3f}) -- \"{snippet}...\"")

    print()
    print("=" * 70)
    print("  Elasticsearch bootstrap complete!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()
