"""
Build the legal knowledge graph from A2AJ documents, checkpointed so
interrupted runs resume.

Design:
  - Document-level extraction (one Gemini call per document, not per
    chunk) — balances graph coverage against Gemini quota.
  - Uses GeminiChat (already wired to gemini_key_manager with 12-key
    rotation + circuit breaker).
  - Checkpoint: processed document ids + incremental graph JSON saved
    every N documents.
  - Rate-limit tolerant: any extraction failure marks the doc for retry
    on the next run (never marked done on failure).

Usage:
    python scripts/build_kg_checkpointed.py [--datasets CHRT,SST] [--max N] [--reset] [--save-every N]
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
    KNOWLEDGE_GRAPH_PATH,
    GRAPHRAG_ENABLED,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "data" / "kg_build.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("kg_build")

from rag_pipeline.a2aj_ingester import load_and_filter_dataset, get_all_datasets
from rag_pipeline.knowledge_graph import LegalKnowledgeGraph

CHECKPOINT_FILE = Path(__file__).parent.parent / "data" / "kg_build_checkpoint.json"


def main():
    parser = argparse.ArgumentParser(description="Checkpointed KG build")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated dataset codes or 'all'")
    parser.add_argument("--max", type=int, default=0,
                        help="Max documents to process (0 = all)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset checkpoint and start fresh")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save graph + checkpoint every N docs")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint reset")

    # Load checkpoint
    processed_ids = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            processed_ids = set(json.load(f).get("processed_ids", []))
        logger.info(f"Resuming with {len(processed_ids)} already-processed documents")

    # Load graph state
    kg = LegalKnowledgeGraph()
    if KNOWLEDGE_GRAPH_PATH.exists():
        kg.load(KNOWLEDGE_GRAPH_PATH)
        logger.info(f"Loaded existing graph: {kg.node_count} nodes, {kg.edge_count} edges")
    else:
        logger.info("No existing graph — starting fresh")

    # Determine datasets
    if args.datasets == "all":
        datasets = get_all_datasets()
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    # Load documents (document-level, not chunked)
    logger.info(f"Loading datasets: {', '.join(datasets)}")
    all_docs = []
    for ds in datasets:
        docs = load_and_filter_dataset(ds, max_cases=0)
        all_docs.extend(docs)
        logger.info(f"  {ds}: {len(docs)} employment documents")
        time.sleep(1)

    logger.info(f"Total documents: {len(all_docs)}")

    # Filter to unprocessed
    to_process = [d for d in all_docs if d.id not in processed_ids]
    logger.info(f"To process: {len(to_process)} (skipping {len(all_docs) - len(to_process)})")

    if args.max:
        to_process = to_process[:args.max]

    if not to_process:
        logger.info("Nothing to do — all documents already extracted")
        kg.save()
        logger.info(f"Final graph: {kg.node_count} nodes, {kg.edge_count} edges")
        return

    # Extract triples per document
    total_triples = 0
    failed = 0
    for i, doc in enumerate(to_process):
        try:
            triples = kg.extract_triples_from_text(doc.content, doc.id)
            for triple in triples:
                kg.add_triple(triple)
                total_triples += 1
            processed_ids.add(doc.id)  # mark done ONLY after successful extraction
        except Exception as e:
            failed += 1
            logger.error(f"  Extraction failed for {doc.id}: {e}")

        if (i + 1) % args.save_every == 0:
            kg.save()
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"processed_ids": list(processed_ids)}, f)
            logger.info(f"  Checkpoint {i+1}/{len(to_process)}: {kg.node_count} nodes, "
                        f"{kg.edge_count} edges, {total_triples} triples, {failed} failed")

    # Final save
    kg.save()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_ids": list(processed_ids)}, f)

    logger.info(f"\n{'='*60}")
    logger.info("KG BUILD COMPLETE")
    logger.info(f"  Processed: {len(processed_ids)} documents")
    logger.info(f"  Triples:   {total_triples}")
    logger.info(f"  Failed:    {failed}")
    logger.info(f"  Graph:     {kg.node_count} nodes, {kg.edge_count} edges")
    logger.info(f"  Saved to:  {KNOWLEDGE_GRAPH_PATH}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()