"""
Enrich existing Pinecone chunks with RAGFlow-style metadata:
  - boost_terms: legal citations/sections/acronyms (pure regex, no LLM)
  - parent_id:   base document id (string op)
  - parent content stored in Elasticsearch (ParentStore)

This is a metadata-only update — vectors are NOT re-embedded, so it is
fast and does not consume Gemini quota. Checkpointed: processed chunk
ids are persisted so interrupted runs resume.

Usage:
    python scripts/enrich_metadata.py [--datasets CHRT,SST] [--max N] [--reset]
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

from rag_pipeline.a2aj_ingester import (
    load_and_filter_dataset,
    chunk_document,
)
from rag_pipeline.legal_document_ingester import (
    derive_parent_id,
    compute_boost_terms,
    store_parent_content,
)
from rag_pipeline.parent_store import ParentStore

CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "enrich_checkpoint.json"


def main():
    parser = argparse.ArgumentParser(description="Enrich Pinecone metadata (boost_terms + parent_id)")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated dataset codes or 'all'")
    parser.add_argument("--max", type=int, default=0,
                        help="Max chunks to process (0 = all)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset checkpoint and start fresh")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint reset")

    # Load checkpoint
    processed_ids = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            processed_ids = set(json.load(f).get("processed_ids", []))
        logger.info(f"Resuming with {len(processed_ids)} already-processed chunks")

    # Determine datasets — default to the 6 employment datasets that exist in
    # Pinecone (matches background_a2aj_embedder). Avoids get_all_datasets(),
    # which lists every HF repo folder including non-ingested tribunals.
    if args.datasets == "all":
        datasets = ["CHRT", "CIRB", "FPSLREB", "OHSTC", "SST", "SCC"]
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Load + chunk all docs (reuses ingester pipeline for identical IDs)
    logger.info(f"Loading datasets: {', '.join(datasets)}")
    all_chunks = []
    for ds in datasets:
        docs = load_and_filter_dataset(ds, max_cases=0)
        for doc in docs:
            all_chunks.extend(chunk_document(doc, max_chunk_size=4000, overlap=200))
        logger.info(f"  {ds}: {len(docs)} docs -> {sum(1 for c in all_chunks if c.source == f'a2aj_{ds.lower()}')} chunks")
        time.sleep(1)

    logger.info(f"Total chunks loaded: {len(all_chunks)}")

    # Filter to unprocessed
    to_process = [c for c in all_chunks if c.id not in processed_ids]
    logger.info(f"To process: {len(to_process)} (skipping {len(all_chunks) - len(to_process)})")

    if args.max:
        to_process = to_process[:args.max]

    if not to_process:
        logger.info("Nothing to do — all chunks already enriched")
        return

    # Connect to Pinecone
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Verify which IDs actually exist in Pinecone (avoid creating empty records)
    logger.info("Fetching existing IDs from Pinecone...")
    existing_ids = set()
    try:
        # list() paginates through all ids in the namespace (limit max 100)
        for ids_batch in index.list(namespace=CHUNK_NAMESPACE, limit=100):
            existing_ids.update(ids_batch.get("vectors", []))
    except Exception as e:
        logger.warning(f"index.list failed ({e}); will attempt update and ignore misses")

    logger.info(f"Existing vectors in namespace: {len(existing_ids)}")

    # Enrich
    parent_store = ParentStore()
    seen_parents = set()
    updated = 0
    skipped = 0
    failed = 0

    for i, chunk in enumerate(to_process):
        try:
            if existing_ids and chunk.id not in existing_ids:
                skipped += 1
                continue

            # Parent content -> ES (once per parent)
            if PARENT_CHILD_ENABLED:
                store_parent_content(chunk, seen_parents)

            # Metadata-only update (no re-embedding)
            metadata = {}
            if KEYWORD_BOOST_ENABLED:
                metadata["boost_terms"] = compute_boost_terms(chunk.content)
            if PARENT_CHILD_ENABLED:
                metadata["parent_id"] = derive_parent_id(chunk.id)

            if metadata:
                index.update(
                    id=chunk.id,
                    set_metadata=metadata,
                    namespace=CHUNK_NAMESPACE,
                )

            processed_ids.add(chunk.id)
            updated += 1

            if (i + 1) % 100 == 0:
                logger.info(f"  Enriched {i+1}/{len(to_process)} ({updated} updated, {skipped} skipped, {failed} failed)")
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump({"processed_ids": list(processed_ids)}, f)

        except Exception as e:
            failed += 1
            logger.error(f"  Failed {chunk.id}: {e}")

    # Save checkpoint
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_ids": list(processed_ids)}, f)

    logger.info(f"\n{'='*60}")
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"  Updated: {updated}")
    logger.info(f"  Skipped (not in Pinecone): {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total processed: {len(processed_ids)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()