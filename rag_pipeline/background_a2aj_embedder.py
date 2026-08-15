"""
Background A2AJ Embedder
========================
Slow-but-steady embedding of already-downloaded A2AJ parquet files.
Runs in the background, processing one document at a time with delays
to respect Gemini free tier rate limits.

Usage:
    python rag_pipeline/background_a2aj_embedder.py [--max N] [--delay D]
    
    --max: Maximum documents to process (default: 5000)
    --delay: Delay between embeddings in seconds (default: 3)
"""

import os, sys, json, time, hashlib, logging, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd

from config import (
    LOG_FORMAT, LOG_LEVEL, GEMINI_API_KEY,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    CHUNK_NAMESPACE, DOCUMENT_SUMMARY_NAMESPACE,
    KEYWORD_BOOST_ENABLED, PARENT_CHILD_ENABLED,
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "data" / "background_embedder.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("bg_embedder")

from rag_pipeline.legal_document_ingester import LegalDocument, chunk_document, generate_embedding
from rag_pipeline.gemini_key_manager import key_manager as km


def load_filtered_parquet(ds_name: str) -> List[LegalDocument]:
    """Load an already-filtered employment law parquet file."""
    path = Path(__file__).parent.parent / "data" / f"a2aj_{ds_name}_employment.parquet"
    if not path.exists():
        logger.warning(f"Parquet not found: {path}")
        return []
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} documents from {ds_name}")
    
    docs = []
    for _, row in df.iterrows():
        name = str(row.get("name_en")) if pd.notna(row.get("name_en")) else "Unknown Case"
        citation = str(row.get("citation_en")) if pd.notna(row.get("citation_en")) else ""
        text = str(row.get("unofficial_text_en")) if pd.notna(row.get("unofficial_text_en")) else ""
        date = row.get("document_date_en") if pd.notna(row.get("document_date_en")) else None
        url = str(row.get("url_en")) if pd.notna(row.get("url_en")) else ""
        ds_code = str(row.get("dataset")) if pd.notna(row.get("dataset")) else ds_name
        
        if len(text.strip()) < 200:
            continue
        
        text = text[:50000]
        
        year = ""
        if date is not None:
            try:
                year = str(pd.Timestamp(date).year)
            except:
                pass
        
        unique_str = f"{citation}_{name}_{year}_{ds_code}"
        doc_id = f"a2aj_{ds_code.lower()}_{hashlib.md5(unique_str.encode()).hexdigest()[:12]}"
        
        docs.append(LegalDocument(
            id=doc_id,
            title=name[:200],
            content=text,
            source=f"a2aj_{ds_code.lower()}",
            case_type=ds_code,
            year=year,
            jurisdiction="Canada",
            court=ds_code,
            citation=citation,
            topic="Employment Law",
            url=url,
        ))
    
    logger.info(f"Converted {len(docs)} docs from {ds_name}")
    return docs


def main():
    parser = argparse.ArgumentParser(description="Background A2AJ Embedder")
    parser.add_argument("--max", type=int, default=5000, help="Max documents to process")
    parser.add_argument("--delay", type=float, default=3.0, help="Delay between embeddings (seconds)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset list (default: CHRT,CIRB,FPSLREB,OHSTC,SST,SCC)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Checkpoint file
    checkpoint_file = Path(__file__).parent.parent / "data" / "bg_embedder_checkpoint.json"
    
    # Datasets to process
    datasets = (args.datasets or "CHRT,CIRB,FPSLREB,OHSTC,SST,SCC").split(",")
    datasets = [d.strip().upper() for d in datasets if d.strip()]
    
    # Load checkpoint if resuming
    processed_ids = set()
    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            cp = json.load(f)
            processed_ids = set(cp.get("processed_ids", []))
            logger.info(f"Resuming with {len(processed_ids)} already-processed IDs")
    
    # Load all docs
    all_docs = []
    for ds in datasets:
        docs = load_filtered_parquet(ds)
        all_docs.extend(docs)
    
    logger.info(f"Total documents: {len(all_docs)}")
    
    # Chunk (simple chunking for large docs)
    chunked_docs = []
    for doc in all_docs:
        chunks = chunk_document(doc, max_chunk_size=4000, overlap=200)
        chunked_docs.extend(chunks)
    
    logger.info(f"After chunking: {len(chunked_docs)} chunks")
    
    # Filter out already-processed
    to_process = [d for d in chunked_docs if d.id not in processed_ids]
    logger.info(f"To process: {len(to_process)} chunks (skipping {len(chunked_docs) - len(to_process)})")
    
    if args.max:
        to_process = to_process[:args.max]
    
    logger.info(f"Will process: {len(to_process)} chunks")
    logger.info(f"Delay: {args.delay}s between embeddings")
    logger.info(f"API keys available: {km.key_count} (auto-rotate on 429)")
    
    # Process
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    from rag_pipeline.legal_document_ingester import derive_parent_id, compute_boost_terms, store_parent_content

    # Probe Elasticsearch once up-front. If it is down, skip parent-content
    # storage entirely instead of paying a connection-timeout on every chunk.
    es_available = False
    if PARENT_CHILD_ENABLED:
        try:
            import urllib.request
            req = urllib.request.urlopen("http://localhost:9200", timeout=3)
            es_available = req.status == 200
        except Exception:
            es_available = False
        logger.info(f"ParentStore: Elasticsearch {'available' if es_available else 'UNAVAILABLE'} "
                    f"(skipping parent-content writes)")

    batch = []
    batch_ids = []
    upserted_count = 0
    failed_count = 0
    seen_parents = set()
    
    for i, doc in enumerate(to_process):
        try:
            # Slow but steady
            time.sleep(args.delay)
            
            emb = generate_embedding(doc.content)
            boost_terms = compute_boost_terms(doc.content)
            metadata = {
                "title": doc.title[:200],
                "case_name": doc.title[:200],
                "content": doc.content[:1000],
                "source": doc.source,
                "case_type": doc.case_type,
                "year": doc.year,
                "jurisdiction": doc.jurisdiction,
                "court": doc.court,
                "citation": doc.citation,
                "topic": doc.topic,
                "url": doc.url,
                "chunk_index": doc.chunk_index,
            }
            if boost_terms:
                metadata["boost_terms"] = boost_terms
            if PARENT_CHILD_ENABLED:
                metadata["parent_id"] = derive_parent_id(doc.id)
            batch.append({
                "id": doc.id,
                "values": emb,
                "metadata": metadata,
            })
            batch_ids.append(doc.id)

            # Store full parent content once per base doc (Elasticsearch)
            if PARENT_CHILD_ENABLED and es_available:
                store_parent_content(doc, seen_parents)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Embedded {i+1}/{len(to_process)} ({upserted_count} upserted, {failed_count} failed)")
            
            # Upsert every 50
            if len(batch) >= 50:
                for attempt in range(3):
                    try:
                        index.upsert(vectors=batch, namespace=CHUNK_NAMESPACE)
                        upserted_count += len(batch)
                        processed_ids.update(batch_ids)  # mark done ONLY after successful upsert
                        logger.info(f"  Upserted {len(batch)} vectors (total {upserted_count})")
                        batch = []
                        batch_ids = []
                        break
                    except Exception as e:
                        logger.warning(f"  Upsert error (attempt {attempt+1}): {e}")
                        time.sleep(5)
                else:
                    failed_count += len(batch)
                    logger.error(f"  Batch upsert failed after 3 retries; will retry on resume (NOT marked done)")
                    batch = []
                    batch_ids = []
            
            # Save checkpoint every 50
            if (i + 1) % 50 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({"processed_ids": list(processed_ids)}, f)
        
        except Exception as e:
            failed_count += 1
            logger.error(f"  Failed doc {doc.id}: {e}")
            # doc not in processed_ids -> retried on resume (no silent loss)
    
    # Final upsert
    if batch:
        try:
            index.upsert(vectors=batch, namespace=CHUNK_NAMESPACE)
            upserted_count += len(batch)
            processed_ids.update(batch_ids)  # mark done ONLY after successful upsert
            logger.info(f"  Final upsert of {len(batch)} vectors")
        except Exception as e:
            logger.error(f"  Final upsert failed: {e}")
            failed_count += len(batch)
            logger.error(f"  Final batch NOT marked done; will retry on resume (no silent loss)")
    
    # Save final checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump({"processed_ids": list(processed_ids), "completed": True}, f)
    
    # Stats
    time.sleep(3)
    stats = index.describe_index_stats()
    
    logger.info(f"\n{'='*60}")
    logger.info("BACKGROUND EMBEDDER COMPLETE")
    logger.info(f"  Processed: {upserted_count}")
    logger.info(f"  Failed:    {failed_count}")
    logger.info(f"  Total vectors: {stats.total_vector_count}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
