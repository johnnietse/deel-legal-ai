#!/usr/bin/env python3
"""
Local Re-Indexing Script (Deployment Phase Item 1)

Runs the SemanticChunker against all legal documents in legal_documents.py
and builds a local BM25 index (serialized to disk). This validates the
full ingestion pipeline without requiring external services.

Usage:
  python scripts/reindex_local.py
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    print("=" * 70)
    print("  DEEL LEGAL AI — Re-Indexing Pipeline (v3.0)")
    print("  SemanticChunker + Local BM25 Index")
    print("=" * 70)
    print()

    start = time.time()

    # ---------------------------------------------------------------
    # Step 1: Load documents from legal_documents.py
    # ---------------------------------------------------------------
    print("📄 Step 1: Loading legal documents...")
    from data.legal_documents import COMPREHENSIVE_LEGAL_DOCUMENTS

    docs = COMPREHENSIVE_LEGAL_DOCUMENTS
    print(f"   Found {len(docs)} legal documents ({sum(len(d['content']) for d in docs):,} chars total)")
    print()

    # ---------------------------------------------------------------
    # Step 2: Run SemanticChunker on every document
    # ---------------------------------------------------------------
    print("✂️  Step 2: Running SemanticChunker...")

    from rag_pipeline.document_processor import SemanticChunker, DocumentChunk

    chunker = SemanticChunker(
        max_chunk_tokens=512,
        min_chunk_tokens=50,
        narrative_target=384,
        structured_target=128,
    )

    all_chunks = []
    stats_by_type = {"narrative": 0, "statute": 0, "list": 0}
    stats_by_section = {}

    for doc in docs:
        chunks = chunker.chunk_document(
            text=doc["content"],
            document_id=doc["id"],
            base_metadata={
                "case_type": doc.get("case_type", ""),
                "year": doc.get("year", ""),
                "jurisdiction": doc.get("jurisdiction", ""),
                "topic": doc.get("topic", ""),
                "citations": doc.get("citations", ""),
                "title": doc.get("title", ""),
            },
        )
        all_chunks.extend(chunks)

        for chunk in chunks:
            ct = chunk.metadata.get("content_type", "narrative")
            stats_by_type[ct] = stats_by_type.get(ct, 0) + 1
            sec = chunk.metadata.get("legal_section", "body")
            stats_by_section[sec] = stats_by_section.get(sec, 0) + 1

    print(f"   Created {len(all_chunks)} semantic chunks from {len(docs)} documents")
    print(f"   Avg chunk size: {sum(c.token_count for c in all_chunks) / max(len(all_chunks), 1):.0f} tokens")
    print()
    print(f"   By content type:")
    for ct, count in sorted(stats_by_type.items(), key=lambda x: -x[1]):
        print(f"     {ct:12s}: {count:4d} chunks")
    print()
    print(f"   By legal section:")
    for sec, count in sorted(stats_by_section.items(), key=lambda x: -x[1]):
        print(f"     {sec:20s}: {count:4d} chunks")
    print()

    # ---------------------------------------------------------------
    # Step 3: Build local BM25 index
    # ---------------------------------------------------------------
    print("🔍 Step 3: Building local BM25 index...")

    from rag_pipeline.hybrid_retriever import BM25Index

    bm25 = BM25Index()

    # Convert DocumentChunks to the dict format BM25Index expects
    chunk_dicts = []
    for chunk in all_chunks:
        d = chunk.to_dict()
        chunk_dicts.append(d)

    bm25.build(chunk_dicts)

    print(f"   BM25 index built with {len(chunk_dicts)} documents, {len(bm25._idf)} unique terms")
    print()

    # ---------------------------------------------------------------
    # Step 4: Validate with test queries
    # ---------------------------------------------------------------
    print("🧪 Step 4: Validating with test queries...")
    print()

    test_queries = [
        "What is the Sagaz test for worker classification?",
        "What are the factors for determining employee vs independent contractor?",
        "Notice period requirements under Ontario ESA",
        "Uber driver gig economy classification",
        "Dependent contractor reasonable notice period",
        "Digital platform workers rights Ontario",
        "Misclassification damages remedies",
        "Working for Workers Act employee presumption",
    ]

    for query in test_queries:
        raw_results = bm25.search(query, top_k=3)
        print(f"   Q: \"{query}\"")
        if raw_results:
            for i, (doc_idx, score) in enumerate(raw_results[:3]):
                doc = chunk_dicts[doc_idx]
                chunk_id = doc.get("chunk_id", doc.get("id", f"idx_{doc_idx}"))
                # Show first 80 chars of content for context
                snippet = doc.get("content", "")[:80].replace("\n", " ").strip()
                print(f"      [{i+1}] {chunk_id} (score: {score:.3f}) — \"{snippet}...\"")
        else:
            print(f"      (no results)")
        print()

    # ---------------------------------------------------------------
    # Step 5: Save index and chunk data to disk
    # ---------------------------------------------------------------
    output_dir = Path("data/semantic_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save chunk metadata
    chunks_path = output_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            d = chunk.to_dict()
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"💾 Saved {len(all_chunks)} chunks to {chunks_path}")

    # Save re-index summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "documents_processed": len(docs),
        "chunks_created": len(all_chunks),
        "avg_tokens_per_chunk": round(sum(c.token_count for c in all_chunks) / max(len(all_chunks), 1), 1),
        "content_type_distribution": stats_by_type,
        "section_distribution": stats_by_section,
        "test_queries_run": len(test_queries),
        "duration_seconds": round(time.time() - start, 2),
    }

    summary_path = output_dir / "reindex_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"📊 Saved re-index summary to {summary_path}")

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print(f"  ✅ RE-INDEXING COMPLETE — {len(all_chunks)} chunks in {elapsed:.1f}s")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
