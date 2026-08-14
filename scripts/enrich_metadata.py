"""
Enrich existing Pinecone chunks with RAGFlow-style metadata:
  - boost_terms: legal citations/sections/acronyms (pure regex, no LLM)
  - parent_id:   base document id (string op)
  - parent content stored in Elasticsearch (ParentStore)

This is a metadata-only update — vectors are NOT re-embedded, so it is
fast and does not consume Gemini quota.

SOURCE OF TRUTH IS PINECONE ITSELF: it iterates the actual vector IDs in
the namespace and enriches each real chunk from its stored content.
It does NOT re-derive chunk IDs from parquet — that caused ID drift when
the HF dataset updated (CI fetched fresh parquet with different IDs than
the local cache that originally populated Pinecone).

Checkpointed: processed chunk ids persist so interrupted runs resume.

Usage:
    python scripts/enrich_metadata.py [--max N] [--reset]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from config import (
    LOG_FORMAT, LOG_LEVEL,
    PINECONE_API_KEY, PINECONE_INDEX_NAME, CHUNK_NAMESPACE,
    KEYWORD_BOOST_ENABLED, PARENT_CHILD_ENABLED,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "data" / "enrich_metadata.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("enrich_metadata")

from rag_pipeline.legal_document_ingester import (
    derive_parent_id,
    compute_boost_terms,
)
from rag_pipeline.parent_store import ParentStore

CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "enrich_checkpoint.json"
FETCH_BATCH = 100
UPDATE_BATCH = 50


def list_all_ids(index) -> list:
    """Return every vector id in the namespace."""
    ids = []
    paginator = index.list(namespace=CHUNK_NAMESPACE, limit=100)
    for page in paginator:
        # SDK returns ListResponse objects with .vectors; older versions
        # returned dicts. Handle both.
        vectors = getattr(page, "vectors", None)
        if vectors is None and isinstance(page, dict):
            vectors = page.get("vectors", [])
        for item in vectors:
            vid = getattr(item, "id", None)
            if vid is None and isinstance(item, dict):
                vid = item.get("id")
            if vid:
                ids.append(vid)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Enrich Pinecone metadata (boost_terms + parent_id)")
    parser.add_argument("--max", type=int, default=0,
                        help="Max chunks to process (0 = all)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset checkpoint and start fresh")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint reset")

    processed_ids = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            processed_ids = set(json.load(f).get("processed_ids", []))
        logger.info(f"Resuming with {len(processed_ids)} already-processed chunks")

    # Connect to Pinecone
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # List all real vector ids in the namespace
    all_ids = list_all_ids(index)
    logger.info(f"Vectors in namespace '{CHUNK_NAMESPACE}': {len(all_ids)}")

    # Keep only a2aj_* chunks (skip legacy ids like '542491_ontario_2024')
    a2aj_ids = [i for i in all_ids if i.startswith("a2aj_")]
    logger.info(f"a2aj chunk ids: {len(a2aj_ids)}")

    to_process = [i for i in a2aj_ids if i not in processed_ids]
    logger.info(f"To process: {len(to_process)} (skipping {len(a2aj_ids) - len(to_process)})")

    if args.max:
        to_process = to_process[:args.max]

    if not to_process:
        logger.info("Nothing to do — all chunks already enriched")
        return

    parent_store = ParentStore()
    updated = 0
    failed = 0
    i = 0

    while i < len(to_process):
        batch_ids = to_process[i:i + FETCH_BATCH]
        try:
            fetch_resp = index.fetch(ids=batch_ids, namespace=CHUNK_NAMESPACE)
            vectors = fetch_resp.get("vectors", {})
        except Exception as e:
            logger.error(f"Batch fetch failed at {i}: {e}")
            failed += len(batch_ids)
            i += FETCH_BATCH
            continue

        # Compute new metadata per vector
        update_list = []
        for vid, vec in vectors.items():
            try:
                metadata = dict(vec.get("metadata", {}))
                content = metadata.get("content", "")
                if KEYWORD_BOOST_ENABLED:
                    bt = compute_boost_terms(content) or []
                    if bt:
                        metadata["boost_terms"] = bt
                if PARENT_CHILD_ENABLED:
                    metadata["parent_id"] = derive_parent_id(vid)
                update_list.append((vid, vec, metadata))
            except Exception as e:
                logger.warning(f"Compute failed for {vid}: {e}")
                failed += 1

        # Upsert preserves embeddings (same values) while updating metadata.
        # v9.1.0 index.update() is single-id only, so use batched upsert.
        for j in range(0, len(update_list), UPDATE_BATCH):
            sub = update_list[j:j + UPDATE_BATCH]
            try:
                index.upsert(
                    vectors=[
                        {
                            "id": vid,
                            "values": vec.get("values"),
                            "metadata": md,
                        }
                        for vid, vec, md in sub
                    ],
                    namespace=CHUNK_NAMESPACE,
                )
                updated += len(sub)
            except Exception as e:
                logger.error(f"Update batch failed: {e}")
                failed += len(sub)

        # Checkpoint every FETCH_BATCH
        processed_ids.update(vectors.keys())
        i += FETCH_BATCH
        logger.info(f"  Enriched {min(i, len(to_process))}/{len(to_process)} "
                    f"({updated} updated, {failed} failed)")
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"processed_ids": list(processed_ids)}, f)

    logger.info(f"\n{'='*60}")
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"  Updated: {updated}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total processed: {len(processed_ids)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()