#!/usr/bin/env python3
"""
CSV Ingestion Wrapper with Progress Tracking & Quota-Aware Retry
=================================================================
Runs the CSV ingestion in batches, tracks progress, handles rate limits,
and resumes automatically on next run.

Usage:
    python csv_ingestion_wrapper.py [--batch-size N] [--max-retries N] [--reset]
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.legal_document_ingester import (
    load_csv_cases, chunk_document, upsert_to_pinecone
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path(__file__).parent / "ingestion_progress.json"
CSV_PATH = Path(__file__).parent.parent / "data" / "employment_cases_large.csv"


def load_progress() -> dict:
    """Load progress from JSON file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "completed_case_ids": [],
        "total_cases": 0,
        "last_run": None,
        "total_upserted": 0,
        "total_failed": 0,
    }


def save_progress(progress: dict):
    """Save progress to JSON file."""
    progress["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def get_all_case_ids() -> list:
    """Get all case IDs from CSV."""
    import csv
    all_cases = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get('Caseid', '')
            if case_id:
                all_cases.append(case_id)
    return all_cases


def get_remaining_cases(progress: dict, max_cases: int = None) -> list:
    """Get list of case IDs that haven't been processed yet."""
    all_cases = get_all_case_ids()
    completed = set(progress.get("completed_case_ids", []))
    remaining = [cid for cid in all_cases if cid not in completed]
    
    if max_cases:
        remaining = remaining[:max_cases]
    
    return remaining


def run_ingestion_batch(case_ids: list) -> dict:
    """Run ingestion for a batch of case IDs directly (no subprocess)."""
    try:
        # Load only the specified cases by filtering during load
        docs = load_csv_cases(case_ids=case_ids)
        
        if not docs:
            return {"success": True, "upserted": 0, "failed": 0}
        
        # Chunk documents
        chunked_docs = []
        for doc in docs:
            chunks = chunk_document(doc)
            chunked_docs.extend(chunks)
        
        # Upsert to Pinecone
        result = upsert_to_pinecone(chunked_docs)
        
        return {
            "success": True,
            "upserted": result.get("upserted", 0),
            "failed": result.get("failed", 0),
        }
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        return {
            "success": False,
            "upserted": 0,
            "failed": len(case_ids),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="CSV Ingestion Wrapper with Progress Tracking")
    parser.add_argument("--batch-size", type=int, default=20, help="Cases per batch")
    parser.add_argument("--max-cases", type=int, default=None, help="Max total cases to process (None = all)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per batch on quota exhaustion")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    args = parser.parse_args()
    
    # Load progress
    progress = load_progress()
    
    if args.reset:
        logger.info("Resetting progress...")
        progress = {
            "completed_case_ids": [],
            "total_cases": 0,
            "last_run": None,
            "total_upserted": 0,
            "total_failed": 0,
        }
        save_progress(progress)
    
    # Get remaining cases
    remaining = get_remaining_cases(progress, args.max_cases)
    total_remaining = len(remaining)
    
    if total_remaining == 0:
        logger.info("All cases already processed!")
        return
    
    logger.info(f"Total cases in CSV: {len(get_all_case_ids())}")
    logger.info(f"Already completed: {len(progress.get('completed_case_ids', []))}")
    logger.info(f"Remaining to process: {total_remaining}")
    logger.info(f"Batch size: {args.batch_size}")
    
    if args.dry_run:
        logger.info("DRY RUN - would process these case IDs:")
        for cid in remaining[:10]:
            logger.info(f"  {cid}")
        if total_remaining > 10:
            logger.info(f"  ... and {total_remaining - 10} more")
        return
    
    # Process in batches
    for i in range(0, total_remaining, args.batch_size):
        batch = remaining[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (total_remaining + args.batch_size - 1) // args.batch_size
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch {batch_num}/{total_batches} ({len(batch)} cases)")
        logger.info(f"{'='*60}")
        
        retry_count = 0
        while retry_count <= args.max_retries:
            result = run_ingestion_batch(batch)
            
            if result["success"] and result.get("upserted", 0) > 0:
                # Mark cases as completed ONLY on success
                progress["completed_case_ids"].extend(batch)
                progress["total_upserted"] += result.get("upserted", 0)
                progress["total_failed"] += result.get("failed", 0)
                save_progress(progress)
                
                logger.info(f"Batch {batch_num} completed: {result.get('upserted', 0)} upserted, {result.get('failed', 0)} failed")
                break
            else:
                retry_count += 1
                error_msg = result.get("error", "").lower()
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    wait_time = min(60 * (2 ** retry_count), 300)  # Max 5 minutes
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {retry_count}/{args.max_retries}...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Batch failed: {result.get('error', 'Unknown error')}")
                    if retry_count >= args.max_retries:
                        logger.error(f"Max retries reached for batch {batch_num}. Skipping.")
                        # Don't mark as completed on permanent failure
                        progress["total_failed"] += len(batch)
                        save_progress(progress)
                    break
        
        # Small delay between batches to be nice to the API
        if i + args.batch_size < total_remaining:
            time.sleep(2)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("INGESTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total upserted: {progress['total_upserted']}")
    logger.info(f"Total failed: {progress['total_failed']}")
    logger.info(f"Completed cases: {len(progress['completed_case_ids'])}")


if __name__ == "__main__":
    main()