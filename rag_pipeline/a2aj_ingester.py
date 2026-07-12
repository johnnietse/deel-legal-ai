"""
A2AJ Canadian Case Law Ingester
================================
Downloads and filters the A2AJ Canadian Case Law dataset from Hugging Face,
then ingests employment-related cases into Pinecone + BM25.

Usage:
    python rag_pipeline/a2aj_ingester.py --datasets CHRT,SST,CIRB --max 500 --upsert
    python rag_pipeline/a2aj_ingester.py --datasets all --max 1000 --upsert
    python rag_pipeline/a2aj_ingester.py --datasets SCC,ONCA,FC,FCA --max 2000 --upsert
    python rag_pipeline/a2aj_ingester.py --list  # List datasets and counts
    python rag_pipeline/a2aj_ingester.py --datasets all-tribunals --upsert  # All employment tribunals
"""

import os, sys, json, time, hashlib, logging, argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
from huggingface_hub import hf_hub_download

from config import (
    LOG_FORMAT, LOG_LEVEL, GEMINI_API_KEY,
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    CHUNK_NAMESPACE, DOCUMENT_SUMMARY_NAMESPACE,
)
from rag_pipeline.legal_document_ingester import (
    LegalDocument, chunk_document, logger
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("a2aj_ingester")

# Employment law keyword set
EMPLOYMENT_KEYWORDS = {
    # Core employment law
    "wrongful dismissal", "constructive dismissal", "termination of employment",
    "reasonable notice", "notice period", "severance", "pay in lieu",
    "employment contract", "employment agreement", "employment standards",
    "employment law", "labour law", "labor law", "labour relations",
    "labour board", "labor board", "union", "collective agreement",
    "unfair labour practice", "unfair labor practice",
    "wrongful termination", "unjust dismissal",
    
    # Workplace rights
    "human rights", "discrimination", "harassment", "workplace harassment",
    "sexual harassment", "poisoned work environment", "reasonable accommodation",
    "duty to accommodate", "accommodation", "undue hardship",
    "workplace safety", "occupational health", "occupational health and safety",
    "workplace injury", "workers compensation", "worker's compensation",
    
    # Employee/employer
    "employee", "employer", "independent contractor", "employee misclassification",
    "worker classification", "employment relationship",
    "employer-employee", "employer and employee",
    
    # Ontario ESA topics
    "employment standards act", "esa 2000", "overtime pay", "minimum wage",
    "public holiday", "vacation pay", "statutory holiday",
    "temporary help", "assignment employee", "wages",
    "termination clause", "enforceable termination",
    
    # Federal
    "canada labour code", "labour code", "canada labour",
    "federal employment", "federally regulated",
    
    # Specific topics
    "non-compete", "non-competition", "non-solicit", "non-solicitation",
    "restrictive covenant", "confidential information",
    "bonus", "stock option", "rsu", "incentive compensation",
    "retirement", "pension", "benefits",
    "layoff", "furlough", "redundancy", "restructuring",
    
    # Notice & damages
    "reasonable notice period", "common law notice",
    "punitive damages", "aggravated damages", "bad faith damages",
    "moral damages", "wallace damages", "wrongful dismissal damages",
    "mitigation", "duty to mitigate",
    
    # Protected leaves
    "maternity leave", "parental leave", "family leave",
    "sick leave", "medical leave", "compassionate care",
    "family status", "caregiver",
    
    # EI/Social
    "employment insurance", "ei benefits", "unemployment",
    "insurable employment", "loss of employment",
}

# Tribunal/jurisdiction keywords
TRIBUNAL_KEYWORDS = {
    "chrt",  # Canadian Human Rights Tribunal
    "cirb",  # Canada Industrial Relations Board
    "fpslreb",  # Federal Public Sector Labour Relations Board
    "ohstc",  # Occupational Health and Safety Tribunal Canada
    "sst",    # Social Security Tribunal
}

# Dataset descriptions
DATASET_INFO = {
    "CHRT": "Canadian Human Rights Tribunal",
    "CIRB": "Canada Industrial Relations Board",
    "FPSLREB": "Federal Public Sector Labour Relations and Employment Board",
    "OHSTC": "Occupational Health and Safety Tribunal Canada",
    "SST": "Social Security Tribunal (EI appeals)",
    "SCC": "Supreme Court of Canada",
    "ONCA": "Ontario Court of Appeal",
    "FC": "Federal Court",
    "FCA": "Federal Court of Appeal",
    "BCCA": "BC Court of Appeal",
    "BCSC": "BC Supreme Court",
    "NSCA": "Nova Scotia Court of Appeal",
    "CMAC": "Court Martial Appeal Court",
}


def get_all_datasets() -> List[str]:
    """Get list of all available dataset codes."""
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files("a2aj/canadian-case-law", repo_type="dataset")
        datasets = sorted(set(
            f.split("/")[0] for f in files if f.endswith(".parquet")
        ))
        return datasets
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        return []


def download_dataset(ds_name: str) -> Optional[pd.DataFrame]:
    """Download a single parquet dataset from Hugging Face."""
    try:
        path = hf_hub_download(
            "a2aj/canadian-case-law", f"{ds_name}/train.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(path)
        logger.info(f"Downloaded {ds_name}: {len(df)} cases ({len(df.columns)} columns)")
        return df
    except Exception as e:
        logger.error(f"Failed to download {ds_name}: {e}")
        return None


def filter_employment_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame for employment/labour law cases using keyword matching."""
    # Build a combined text field for searching
    search_texts = []
    for _, row in df.iterrows():
        parts = []
        name = row.get("name_en")
        name = str(name) if pd.notna(name) else ""
        citation = row.get("citation_en")
        citation = str(citation) if pd.notna(citation) else ""
        text = row.get("unofficial_text_en")
        text = str(text)[:2000] if pd.notna(text) else ""
        parts.append(name.lower())
        parts.append(citation.lower())
        if text:
            parts.append(text[:2000].lower())
        search_texts.append(" ".join(parts))
    
    # Keyword matching
    mask = pd.Series(False, index=df.index)
    for i, text in enumerate(search_texts):
        for kw in EMPLOYMENT_KEYWORDS:
            if kw in text:
                mask.iloc[i] = True
                break
        if not mask.iloc[i]:
            # Check tribunal names in dataset
            ds_name = str(df.iloc[i].get("dataset", "")) if pd.notna(df.iloc[i].get("dataset")) else ""
            if ds_name.lower() in TRIBUNAL_KEYWORDS:
                mask.iloc[i] = True
    
    matches = df[mask].copy()
    logger.info(f"Filtered {len(df)} -> {len(matches)} employment-related cases "
                f"({100*len(matches)/max(len(df),1):.1f}%)")
    return matches


def convert_to_legal_documents(
    df: pd.DataFrame, source_label: str, max_docs: Optional[int] = None
) -> List[LegalDocument]:
    """Convert A2AJ DataFrame rows to LegalDocument objects."""
    docs = []
    count = 0
    
    for _, row in df.iterrows():
        if max_docs and count >= max_docs:
            break
        
        name = str(row.get("name_en")) if pd.notna(row.get("name_en")) else "Unknown Case"
        citation = str(row.get("citation_en")) if pd.notna(row.get("citation_en")) else ""
        text = str(row.get("unofficial_text_en")) if pd.notna(row.get("unofficial_text_en")) else ""
        date = row.get("document_date_en")
        url = str(row.get("url_en")) if pd.notna(row.get("url_en")) else ""
        ds_name = str(row.get("dataset")) if pd.notna(row.get("dataset")) else source_label
        
        # Skip if no real content
        if len(text.strip()) < 200:
            continue
        
        # Truncate very long texts
        if len(text) > 50000:
            text = text[:50000]
        
        # Extract year from date
        year = ""
        if pd.notna(date):
            try:
                year = str(pd.Timestamp(date).year)
            except:
                pass
        
        # Determine court/tribunal
        court_map = {
            "SCC": "Supreme Court of Canada",
            "FC": "Federal Court",
            "FCA": "Federal Court of Appeal",
            "ONCA": "Ontario Court of Appeal",
            "BCCA": "BC Court of Appeal",
            "BCSC": "BC Supreme Court",
            "CHRT": "Canadian Human Rights Tribunal",
            "CIRB": "Canada Industrial Relations Board",
            "FPSLREB": "Federal Public Sector Labour Relations and Employment Board",
            "OHSTC": "Occupational Health and Safety Tribunal Canada",
            "SST": "Social Security Tribunal",
            "NSCA": "Nova Scotia Court of Appeal",
            "CMAC": "Court Martial Appeal Court",
        }
        court = court_map.get(ds_name, ds_name)
        
        # Set case type based on source
        case_type_map = {
            "SCC": "Supreme Court of Canada",
            "FC": "Federal Court",
            "FCA": "Federal Court of Appeal",
            "ONCA": "Ontario Court of Appeal",
            "BCCA": "BC Court of Appeal",
            "BCSC": "BC Supreme Court",
            "CHRT": "Human Rights Tribunal",
            "CIRB": "Industrial Relations Board",
            "FPSLREB": "Federal Public Sector Labour Relations",
            "OHSTC": "Occupational Health and Safety",
            "SST": "Social Security Tribunal",
        }
        case_type = case_type_map.get(ds_name, "Canadian Legal Case")
        
        # Determine jurisdiction
        juris_map = {
            "SCC": "Canada (Federal)",
            "FC": "Canada (Federal)",
            "FCA": "Canada (Federal)",
            "ONCA": "Ontario",
            "BCCA": "British Columbia",
            "BCSC": "British Columbia",
            "CHRT": "Canada (Federal)",
            "CIRB": "Canada (Federal)",
            "FPSLREB": "Canada (Federal)",
            "OHSTC": "Canada (Federal)",
            "SST": "Canada (Federal)",
            "NSCA": "Nova Scotia",
            "CMAC": "Canada (Federal)",
        }
        jurisdiction = juris_map.get(ds_name, "Canada")
        
        # Determine topic
        # Try to infer from content
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["human right", "discrimination", "harassment"]):
            topic = "Human Rights & Discrimination"
        elif any(kw in text_lower for kw in ["union", "collective agreement", "labour relation", "certification"]):
            topic = "Labour Relations & Unions"
        elif any(kw in text_lower for kw in ["wrongful dismissal", "constructive dismissal", "termination", "notice"]):
            topic = "Wrongful Dismissal & Termination"
        elif any(kw in text_lower for kw in ["ei", "employment insurance", "unemployment"]):
            topic = "Employment Insurance"
        elif any(kw in text_lower for kw in ["worker compensation", "workplace injur", "occupational health"]):
            topic = "Workplace Safety & Compensation"
        elif any(kw in text_lower for kw in ["independent contractor", "employee misclassification"]):
            topic = "Worker Classification"
        elif any(kw in text_lower for kw in ["employment standard", "esa", "overtime", "minimum wage"]):
            topic = "Employment Standards"
        elif any(kw in text_lower for kw in ["accommodation", "family status", "caregiver"]):
            topic = "Accommodation & Leave"
        elif any(kw in text_lower for kw in ["non-compete", "non-solicit", "restrictive covenant"]):
            topic = "Restrictive Covenants"
        else:
            topic = "Employment Law"
        
        # Create unique ID from citation + hash
        unique_str = f"{citation}_{name}_{year}_{court}"
        doc_id = f"a2aj_{ds_name.lower()}_{hashlib.md5(unique_str.encode()).hexdigest()[:12]}"
        
        doc = LegalDocument(
            id=doc_id,
            title=name[:200],
            content=text,
            source=f"a2aj_{ds_name.lower()}",
            case_type=case_type,
            year=year,
            jurisdiction=jurisdiction,
            court=court,
            citation=citation,
            topic=topic,
            url=url,
        )
        docs.append(doc)
        count += 1
    
    logger.info(f"Converted {len(docs)} A2AJ cases from {source_label}")
    return docs


def load_and_filter_dataset(
    ds_name: str, max_cases: Optional[int] = None
) -> List[LegalDocument]:
    """End-to-end: download -> filter -> convert for one dataset."""
    df = download_dataset(ds_name)
    if df is None or len(df) == 0:
        return []
    
    matches = filter_employment_cases(df)
    if len(matches) == 0:
        logger.info(f"No employment cases found in {ds_name}")
        return []
    
    docs = convert_to_legal_documents(matches, ds_name, max_docs=max_cases)
    return docs


# ---------------------------------------------------------------------------
# Batch Embedding (much faster than per-document embedding)
# ---------------------------------------------------------------------------

def batch_generate_embeddings(
    texts: List[str],
    batch_size: int = 10,
    max_retries: int = 5,
) -> Dict[str, List[float]]:
    """
    Generate embeddings in batches using Gemini batchEmbedContents API.
    Much faster than calling embedContent per document.
    
    Returns dict mapping text -> embedding vector.
    """
    import requests as http_req
    
    result = {}
    pending = [(i, t) for i, t in enumerate(texts) if t.strip()]
    
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_texts = [t for _, t in batch]
        
        # Build batch request
        requests_list = []
        for t in batch_texts:
            requests_list.append({
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": t[:8000]}]},
            })
        
        for attempt in range(max_retries):
            try:
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-embedding-001:batchEmbedContents?key={GEMINI_API_KEY}"
                )
                resp = http_req.post(url, json={"requests": requests_list}, timeout=60)
                
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    for (idx_in_batch, orig_idx, _), emb in zip(batch, embeddings):
                        result[orig_idx] = emb.get("values", [])
                    logger.info(f"  Batch {batch_start//batch_size + 1}/"
                                f"{max(1, (len(pending) + batch_size - 1)//batch_size)}: "
                                f"embedded {len(batch)} texts (total {len(result)})")
                    break
                elif resp.status_code == 429:
                    retry_after = 45
                    try:
                        err_data = resp.json()
                        if "error" in err_data and "details" in err_data.get("error", {}):
                            for d in err_data["error"]["details"]:
                                if d.get("@type", "").endswith("RetryInfo"):
                                    retry_after = int(d.get("retryDelay", "45s").rstrip("s"))
                    except:
                        pass
                    jitter = random.uniform(0, 2)
                    delay = retry_after + jitter
                    logger.warning(f"  Batch rate limited (attempt {attempt+1}/{max_retries}), "
                                   f"waiting {delay:.0f}s...")
                    time.sleep(delay)
                    # Use single embedding for retry (batch might be too heavy)
                    for orig_idx, t in batch:
                        try:
                            emb = _embed_single(t)
                            result[orig_idx] = emb
                            logger.info(f"  Single embed {orig_idx} after batch retry")
                        except Exception as e:
                            logger.error(f"  Single embed failed for {orig_idx}: {e}")
                    break
                else:
                    logger.warning(f"  Batch API error {resp.status_code}, falling back to single...")
                    for orig_idx, t in batch:
                        try:
                            emb = _embed_single(t)
                            result[orig_idx] = emb
                        except Exception as e:
                            logger.error(f"  Single embed failed for {orig_idx}: {e}")
                    break
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"  Batch error (attempt {attempt+1}): {e}, retrying...")
                    time.sleep(5)
                else:
                    logger.error(f"  Batch failed after {max_retries} retries: {e}")
                    for orig_idx, t in batch:
                        try:
                            emb = _embed_single(t)
                            result[orig_idx] = emb
                        except:
                            pass
        
        # Small delay between batches to avoid rate limits
        time.sleep(0.3)
    
    return result


def _embed_single(text: str, max_retries: int = 5) -> List[float]:
    """Embed a single text as fallback."""
    import requests as http_req
    import random
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
    
    for attempt in range(max_retries):
        try:
            resp = http_req.post(url, json={
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text[:8000]}]},
            }, timeout=30)
            
            if resp.status_code == 200:
                return resp.json()["embedding"]["values"]
            elif resp.status_code == 429:
                delay = 45 + random.uniform(0, 2)
                logger.warning(f"  Rate limited, waiting {delay:.0f}s...")
                time.sleep(delay)
                continue
            else:
                raise Exception(f"API error: {resp.status_code}")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    
    raise Exception("Failed to embed after retries")


def fast_upsert_to_pinecone(documents: List[LegalDocument]) -> Dict[str, Any]:
    """
    Faster upsert using batch embeddings and parallel processing.
    """
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Collect all unique texts for embedding
    doc_texts = []
    for doc in documents:
        doc_texts.append(doc.content)
    
    # Batch embed all texts at once
    logger.info(f"Batch embedding {len(doc_texts)} texts...")
    embeddings = batch_generate_embeddings(doc_texts, batch_size=10)
    logger.info(f"Embedded {len(embeddings)}/{len(doc_texts)} texts successfully")
    
    # Build vectors for Pinecone
    vectors_to_upsert = []
    failed = 0
    for i, doc in enumerate(documents):
        if i in embeddings and embeddings[i]:
            vectors_to_upsert.append({
                "id": doc.id,
                "values": embeddings[i],
                "metadata": {
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
                },
            })
        else:
            failed += 1
    
    # Upsert in batches of 50
    logger.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
    for batch_start in range(0, len(vectors_to_upsert), 50):
        batch = vectors_to_upsert[batch_start:batch_start + 50]
        try:
            index.upsert(vectors=batch, namespace=CHUNK_NAMESPACE)
            logger.info(f"  Upserted batch {batch_start//50 + 1}: {len(batch)} vectors")
        except Exception as e:
            logger.error(f"  Upsert batch failed: {e}")
            # Try one by one
            for v in batch:
                try:
                    index.upsert(vectors=[v], namespace=CHUNK_NAMESPACE)
                except:
                    failed += 1
        time.sleep(0.5)
    
    # Document summaries for multi-granularity
    doc_summaries = []
    seen_docs = set()
    for doc in documents:
        parent_id = doc.id.split("_chunk")[0]
        if parent_id not in seen_docs and doc.chunk_index == 1:
            doc_summaries.append(doc)
            seen_docs.add(parent_id)
    
    if doc_summaries:
        logger.info(f"Generating {len(doc_summaries)} document summaries...")
        summary_texts = [d.content[:1024] for d in doc_summaries]
        summary_embs = batch_generate_embeddings(summary_texts, batch_size=10)
        summary_vectors = []
        for i, doc in enumerate(doc_summaries):
            if i in summary_embs and summary_embs[i]:
                summary_vectors.append({
                    "id": f"doc_summary_{doc.id.split('_chunk')[0]}",
                    "values": summary_embs[i],
                    "metadata": {
                        "title": doc.title[:200],
                        "case_name": doc.title[:200],
                        "content": doc.content[:1024],
                        "source": doc.source,
                        "case_type": doc.case_type,
                        "year": doc.year,
                        "jurisdiction": doc.jurisdiction,
                        "topic": doc.topic,
                        "granularity": "document",
                    },
                })
        
        if summary_vectors:
            for batch_start in range(0, len(summary_vectors), 50):
                batch = summary_vectors[batch_start:batch_start + 50]
                try:
                    index.upsert(vectors=batch, namespace=DOCUMENT_SUMMARY_NAMESPACE)
                    logger.info(f"  Upserted {len(batch)} summaries")
                except:
                    pass
                time.sleep(0.3)
    
    # Stats
    time.sleep(3)
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")
    
    # Build BM25
    logger.info("Building BM25 index for hybrid search...")
    try:
        from rag_pipeline.hybrid_retriever import BM25Index
        bm25 = BM25Index()
        bm25_chunks = []
        for doc in documents:
            bm25_chunks.append({
                "id": doc.id,
                "chunk_id": doc.id,
                "content": doc.content,
                "metadata": {
                    "title": doc.title[:200],
                    "case_name": doc.title[:200],
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
            })
        bm25.build(bm25_chunks)
        logger.info(f"BM25 index built with {len(bm25_chunks)} chunks")
    except Exception as e:
        logger.warning(f"BM25 index build failed (non-fatal): {e}")
    
    return {
        "upserted": len(vectors_to_upsert),
        "failed": failed,
        "total_vectors": stats.total_vector_count,
        "namespaces": {k: v.vector_count for k, v in stats.namespaces.items()},
    }


def list_datasets():
    """Print available datasets and their sizes."""
    datasets = get_all_datasets()
    print(f"\n{'Dataset':<12} {'Description':<55} {'Status':<12}")
    print("-" * 80)
    for ds in sorted(datasets):
        desc = DATASET_INFO.get(ds, "")
        # Try to get size without full download
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            meta = api.dataset_info("a2aj/canadian-case-law")
            sizes = {}
            for f in meta.siblings:
                if f.rfilename.endswith(".parquet"):
                    parts = f.rfilename.split("/")
                    if parts[0] == ds:
                        sizes[ds] = f.size
            size_str = ""
            if ds in sizes:
                mb = sizes[ds] / (1024 * 1024)
                size_str = f" ({mb:.0f} MB)"
        except:
            size_str = ""
        
        print(f"{ds:<12} {desc:<55} available{size_str}")
    print(f"\nTotal datasets: {len(datasets)}")
    print("\nDatasets most relevant for employment law:")
    emp_datasets = ["CHRT", "CIRB", "FPSLREB", "OHSTC", "SST", "SCC", "ONCA", "FC", "FCA", "BCCA", "BCSC"]
    for ds in emp_datasets:
        status = "[OK]" if ds in datasets else "[--]"
        print(f"  {status} {ds:<8} {DATASET_INFO.get(ds, '')}")


def main():
    parser = argparse.ArgumentParser(description="A2AJ Canadian Case Law Ingester")
    parser.add_argument("--datasets", type=str, default="all",
                       help="Comma-separated dataset codes or 'all'")
    parser.add_argument("--max", type=int, default=500,
                       help="Max cases to ingest per dataset")
    parser.add_argument("--upsert", action="store_true",
                       help="Actually upsert to Pinecone")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be ingested without upserting")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets and exit")
    parser.add_argument("--total-max", type=int, default=None,
                       help="Total max documents across all datasets")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    # Determine which datasets to process
    if args.datasets == "all":
        datasets_to_process = get_all_datasets()
    else:
        datasets_to_process = [d.strip() for d in args.datasets.split(",")]
    
    # Prioritize employment-relevant datasets
    priority_order = ["CHRT", "CIRB", "FPSLREB", "OHSTC", "SST", "SCC", "ONCA", "FC", "FCA",
                      "BCCA", "BCSC", "NSCA", "CMAC", "NSSC", "NSPC", "NSFC", "NSSM",
                      "YKCA", "RAD", "RPD", "RLLR", "PSDPT", "OIC", "CT", "CITT", "TCC"]
    datasets_to_process.sort(key=lambda d: priority_order.index(d) if d in priority_order else 999)
    
    logger.info(f"{'='*60}")
    logger.info("A2AJ CANADIAN CASE LAW INGESTION")
    logger.info(f"{'='*60}")
    logger.info(f"Datasets: {', '.join(datasets_to_process)}")
    logger.info(f"Max per dataset: {args.max}")
    
    all_docs: List[LegalDocument] = []
    total_max = args.total_max or (args.max * len(datasets_to_process))
    
    for ds_name in datasets_to_process:
        if total_max and len(all_docs) >= total_max:
            logger.info(f"Reached total max ({total_max}), stopping")
            break
        
        remaining = (total_max - len(all_docs)) if total_max else None
        max_for_this = min(args.max, remaining) if remaining else args.max
        
        logger.info(f"\n--- Processing {ds_name} ---")
        docs = load_and_filter_dataset(ds_name, max_cases=max_for_this)
        logger.info(f"  {ds_name}: {len(docs)} employment documents")
        all_docs.extend(docs)
        
        # Be nice to HF API
        time.sleep(1)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL: {len(all_docs)} employment law documents from A2AJ")
    
    if len(all_docs) == 0:
        logger.warning("No documents to ingest. Try different datasets.")
        return
    
    # Chunk all documents
    logger.info("Chunking documents...")
    chunked_docs = []
    for doc in all_docs:
        chunks = chunk_document(doc, max_chunk_size=4000, overlap=200)
        chunked_docs.extend(chunks)
    logger.info(f"After chunking: {len(chunked_docs)} chunks from {len(all_docs)} documents")
    
    # Topic distribution
    topic_counts = {}
    for doc in all_docs:
        topic_counts[doc.topic] = topic_counts.get(doc.topic, 0) + 1
    logger.info("Topic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {topic}: {count}")
    
    # Source distribution
    source_counts = {}
    for doc in all_docs:
        src = doc.source
        source_counts[src] = source_counts.get(src, 0) + 1
    logger.info("Source distribution:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {src}: {count}")
    
    if args.dry_run:
        logger.info("\nDRY RUN - First 10 documents:")
        for doc in chunked_docs[:10]:
            logger.info(f"  [{doc.source}] {doc.title} ({len(doc.content)} chars)")
        if len(chunked_docs) > 10:
            logger.info(f"  ... and {len(chunked_docs) - 10} more")
        logger.info(f"\nTotal: {len(chunked_docs)} chunks ready for upsert")
        return
    
    if args.upsert:
        logger.info(f"\nUpserting {len(chunked_docs)} chunks to Pinecone via batch embedding...")
        result = fast_upsert_to_pinecone(chunked_docs)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info(f"  Documents ingested: {len(all_docs)}")
        logger.info(f"  Chunks upserted:   {result.get('upserted', 0)}")
        logger.info(f"  Failed:             {result.get('failed', 0)}")
        logger.info(f"  Total vectors:      {result.get('total_vectors', '?')}")
        logger.info(f"  Namespaces:         {result.get('namespaces', {})}")
        logger.info("=" * 60)
    else:
        logger.info(f"\nReady to upsert {len(chunked_docs)} chunks")
        logger.info("Use --upsert to actually push to Pinecone")


if __name__ == "__main__":
    main()
