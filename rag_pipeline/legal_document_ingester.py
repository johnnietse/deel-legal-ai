"""
Legal Document Ingestion Pipeline — CanLII + Additional Sources
===============================================================
Unified script for scraping, downloading, parsing, embedding, and
upserting legal documents into Pinecone (and optionally Milvus).

Sources:
  1. CanLII — Canadian legal cases (PDF scraping + metadata extraction)
  2. Curated legal documents (from data/legal_documents.py)
  3. Supplementary legal concepts (classifications, factors, tests)
  4. Real case law from CanLII.org (downloadable PDFs/HTML)
  5. Federal/provincial legislation (ESA, CLC, OHSA, etc.)

Usage:
    python rag_pipeline/legal_document_ingester.py --source canlii --max 50
    python rag_pipeline/legal_document_ingester.py --source curated
    python rag_pipeline/legal_document_ingester.py --source all --upsert
"""

import os
import sys
import json
import re
import time
import logging
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field, asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME,
    DATA_DIR, CHUNK_NAMESPACE, DOCUMENT_SUMMARY_NAMESPACE,
    LOG_FORMAT, LOG_LEVEL,
    KEYWORD_BOOST_ENABLED, PARENT_CHILD_ENABLED,
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LegalDocument:
    """A legal document ready for embedding and indexing."""
    id: str
    title: str
    content: str
    source: str = "canlii"  # canlii, curated, legislation, concept
    case_type: str = "Unknown"
    year: str = ""
    jurisdiction: str = "Canada"
    court: str = ""
    citation: str = ""
    topic: str = "Employment Law"
    url: str = ""
    chunk_index: int = 0
    doc_level: str = "chunk"  # "chunk" or "summary"


# ---------------------------------------------------------------------------
# Source 1: Curated Documents (from data/legal_documents.py)
# ---------------------------------------------------------------------------

def load_curated_documents() -> List[LegalDocument]:
    """Load the 100+ curated legal documents from the codebase."""
    from data.legal_documents import ALL_LEGAL_DOCUMENTS
    docs = []
    for d in ALL_LEGAL_DOCUMENTS:
        docs.append(LegalDocument(
            id=d["id"],
            title=d["title"],
            content=d["content"],
            source="curated",
            case_type=d.get("case_type", "Unknown"),
            year=d.get("year", ""),
            jurisdiction=d.get("jurisdiction", "Canada"),
            citation=d.get("citations", ""),
            topic=d.get("topic", "Employment Law"),
        ))
    logger.info(f"Loaded {len(docs)} curated legal documents")
    return docs


# ---------------------------------------------------------------------------
# Source 2: CanLII Scraper Integration
# ---------------------------------------------------------------------------

def scrape_canlii_cases(max_cases: int = 20) -> List[LegalDocument]:
    """
    Scrape employment law cases from CanLII (the existing scraper).
    Returns LegalDocument objects extracted from scraped PDFs.
    """
    from rag_pipeline.canlii_scraper import CanLIIScraper

    scraper = CanLIIScraper()
    results = scraper.scrape_employment_cases(max_cases=max_cases)
    
    docs = []
    success_count = 0
    for r in results:
        if r.status == "success" and r.pdf_path:
            try:
                # Extract text from PDF
                content = _extract_pdf_text(r.pdf_path)
                if content and len(content.strip()) > 100:
                    doc_id = f"canlii_{r.case_id}_{hashlib.md5(content[:100].encode()).hexdigest()[:8]}"
                    docs.append(LegalDocument(
                        id=doc_id,
                        title=r.case_name or f"CanLII Case {r.case_id}",
                        content=content[:15000],  # Limit content size
                        source="canlii",
                        case_type="CanLII Case",
                        year=r.date[:4] if r.date else "",
                        jurisdiction=r.jurisdiction or "Canada",
                        court=r.court or "",
                        citation=r.citation or "",
                        topic="Employment Law",
                        url=r.url,
                    ))
                    success_count += 1
            except Exception as e:
                logger.warning(f"Failed to extract text from {r.pdf_path}: {e}")
    
    logger.info(f"Scraped {success_count} CanLII cases successfully")
    return docs


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF file using available libraries."""
    content = ""
    try:
        # Try PyMuPDF first (fastest)
        import fitz
        doc = fitz.open(pdf_path)
        for page in doc:
            content += page.get_text()
        doc.close()
    except ImportError:
        try:
            # Fall back to pdfplumber
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
        except ImportError:
            logger.error("No PDF extraction library available (install PyMuPDF or pdfplumber)")
    return content


# ---------------------------------------------------------------------------
# Source 3: CanLII.org Web Search (download real case HTML)
# ---------------------------------------------------------------------------

CANLII_BASE = "https://www.canlii.org"
CANLII_SEARCH_URL = f"{CANLII_BASE}/en/on/onsc/search"

def search_canlii_web(
    query: str = "employment law",
    jurisdiction: str = "on",
    max_results: int = 20,
) -> List[LegalDocument]:
    """
    Search CanLII.org for real employment law cases and download their content.
    Fetches HTML versions (not PDF) for easier text extraction.
    
    This is the primary method for getting REAL case law into the vector DB.
    """
    import requests
    from bs4 import BeautifulSoup
    
    docs = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
    })
    
    try:
        # Perform search
        search_url = f"{CANLII_BASE}/en/{jurisdiction}/search"
        params = {
            "q": query,
            "sort": "decisionDateDesc",
            "maxResults": min(max_results, 50),
        }
        
        logger.info(f"Searching CanLII: {search_url}?q={query}")
        resp = session.get(search_url, params=params, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Extract case links
        case_links = []
        for link in soup.select("a[href*='/en/']"):
            href = link.get("href", "")
            if "/en/" in href and "/judgment/" in href:
                case_links.append(href)
        
        # Deduplicate
        case_links = list(dict.fromkeys(case_links))
        logger.info(f"Found {len(case_links)} case links")
        
        # Download each case
        for i, case_path in enumerate(case_links[:max_results]):
            try:
                case_url = f"{CANLII_BASE}{case_path}"
                logger.info(f"Downloading case {i+1}/{len(case_links[:max_results])}: {case_url}")
                
                resp = session.get(case_url, timeout=30)
                resp.raise_for_status()
                case_soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extract case title
                title_elem = case_soup.find("h1") or case_soup.find("title")
                title = title_elem.get_text(strip=True) if title_elem else f"CanLII Case {i}"
                
                # Extract case content (main body)
                content_elem = case_soup.find("div", class_="judgment-content") or \
                               case_soup.find("div", class_="case-content") or \
                               case_soup.find("article") or \
                               case_soup.find("main")
                
                content = content_elem.get_text("\n", strip=True) if content_elem else ""
                
                # Extract metadata
                meta = {}
                for meta_row in case_soup.select("div.meta-row, .case-meta li"):
                    text = meta_row.get_text(" ", strip=True)
                    if ":" in text:
                        key, val = text.split(":", 1)
                        meta[key.strip().lower()] = val.strip()
                
                if content and len(content) > 200:
                    doc_id = f"canlii_web_{i}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
                    docs.append(LegalDocument(
                        id=doc_id,
                        title=title[:200],
                        content=content[:15000],
                        source="canlii_web",
                        case_type="CanLII Case",
                        year=meta.get("date", "")[:4] if meta.get("date") else "",
                        jurisdiction=jurisdiction.upper(),
                        court=meta.get("court", ""),
                        citation=meta.get("citation", ""),
                        topic="Employment Law",
                        url=case_url,
                    ))
                
                # Rate limiting
                time.sleep(5)
                
            except Exception as e:
                logger.warning(f"Failed to download case {i}: {e}")
                continue
    
    except Exception as e:
        logger.error(f"CanLII search failed: {e}")
    
    logger.info(f"Downloaded {len(docs)} real cases from CanLII")
    return docs


# ---------------------------------------------------------------------------
# Source 4: Additional Legal Sources (credible law websites)
# ---------------------------------------------------------------------------

def scrape_legal_sources() -> List[LegalDocument]:
    """
    Scrape additional credible legal sources for Canadian employment law.
    
    Sources:
    - Supreme Court of Canada judgments (scc-csc.ca)
    - Federal Court judgments (fct-cf.gc.ca)
    - Ontario Court of Appeal (ontariocourts.ca)
    - Canadian Legal Information Institute (canlii.org)
    - Employment and Social Development Canada (canada.ca)
    """
    docs = []
    sources = [
        {
            "name": "Supreme Court of Canada",
            "url": "https://decisions.scc-csc.ca/scc-csc/en/nav.do",
            "case_type": "Supreme Court of Canada",
        },
        {
            "name": "Federal Court of Canada",
            "url": "https://decisions.fct-cf.gc.ca/fc-cf/en/nav.do",
            "case_type": "Federal Court",
        },
    ]
    
    for source in sources:
        try:
            logger.info(f"Scraping {source['name']}...")
            import requests
            from bs4 import BeautifulSoup
            
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            })
            
            resp = session.get(source["url"], timeout=30)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extract case links
                for link in soup.select("a[href*='decision']") or soup.select("a[href*='case']"):
                    href = link.get("href", "")
                    if href and not href.startswith("http"):
                        href = requests.compat.urljoin(source["url"], href)
                    
                    case_name = link.get_text(strip=True)
                    if case_name and len(case_name) > 20:
                        logger.info(f"  Found case: {case_name[:60]}...")
            
            time.sleep(3)
        
        except Exception as e:
            logger.warning(f"Failed to scrape {source['name']}: {e}")
    
    return docs


# ---------------------------------------------------------------------------
# Source 5: Legislation Scraper
# ---------------------------------------------------------------------------

LEGISLATION_SOURCES = [
    {
        "name": "Employment Standards Act, 2000 (Ontario)",
        "url": "https://www.ontario.ca/laws/statute/00e41",
        "jurisdiction": "Ontario",
        "type": "Provincial Statute",
    },
    {
        "name": "Canada Labour Code",
        "url": "https://laws-lois.justice.gc.ca/eng/acts/l-2/",
        "jurisdiction": "Canada (Federal)",
        "type": "Federal Statute",
    },
    {
        "name": "Occupational Health and Safety Act (Ontario)",
        "url": "https://www.ontario.ca/laws/statute/90o01",
        "jurisdiction": "Ontario",
        "type": "Provincial Statute",
    },
    {
        "name": "Pay Equity Act (Ontario)",
        "url": "https://www.ontario.ca/laws/statute/90p07",
        "jurisdiction": "Ontario",
        "type": "Provincial Statute",
    },
    {
        "name": "Human Rights Code (Ontario)",
        "url": "https://www.ontario.ca/laws/statute/90h19",
        "jurisdiction": "Ontario",
        "type": "Provincial Statute",
    },
    {
        "name": "Workplace Safety and Insurance Act (Ontario)",
        "url": "https://www.ontario.ca/laws/statute/97w16",
        "jurisdiction": "Ontario",
        "type": "Provincial Statute",
    },
]


def scrape_legislation() -> List[LegalDocument]:
    """Scrape Canadian employment legislation from official sources."""
    import requests
    from bs4 import BeautifulSoup
    
    docs = []
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    
    for source in LEGISLATION_SOURCES:
        try:
            logger.info(f"Scraping legislation: {source['name']}")
            resp = session.get(source["url"], timeout=30)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Extract main content
            content_elem = soup.find("div", class_="content") or \
                           soup.find("div", class_="layout-content") or \
                           soup.find("article") or \
                           soup.find("main")
            
            content = content_elem.get_text("\n", strip=True) if content_elem else resp.text[:50000]
            
            if content and len(content) > 500:
                doc_id = f"legislation_{source['name'].lower().replace(' ','_').replace(',','').replace('(','').replace(')','')[:40]}"
                docs.append(LegalDocument(
                    id=doc_id,
                    title=source["name"],
                    content=content[:20000],
                    source="legislation",
                    case_type=source["type"],
                    year=datetime.now().year,
                    jurisdiction=source["jurisdiction"],
                    citation=source["url"],
                    topic="Employment Legislation",
                    url=source["url"],
                ))
                logger.info(f"  Extracted {len(content)} chars")
            
            time.sleep(3)
        
        except Exception as e:
            logger.warning(f"Failed to scrape {source['name']}: {e}")
    
    logger.info(f"Scraped {len(docs)} legislation documents")
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(
    doc: LegalDocument,
    max_chunk_size: int = 4000,
    overlap: int = 200,
) -> List[LegalDocument]:
    """Split a large document into smaller chunks for embedding."""
    if len(doc.content) <= max_chunk_size:
        doc.chunk_index = 0
        return [doc]
    
    chunks = []
    paragraphs = doc.content.split("\n\n")
    current_chunk = ""
    chunk_num = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunk_num += 1
                chunk_doc = LegalDocument(
                    id=f"{doc.id}_chunk{chunk_num}",
                    title=f"{doc.title} (Part {chunk_num})",
                    content=current_chunk.strip(),
                    source=doc.source,
                    case_type=doc.case_type,
                    year=doc.year,
                    jurisdiction=doc.jurisdiction,
                    court=doc.court,
                    citation=doc.citation,
                    topic=doc.topic,
                    url=doc.url,
                    chunk_index=chunk_num,
                )
                chunks.append(chunk_doc)
            
            # Keep last part for overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
            current_chunk = overlap_text + para + "\n\n"
    
    # Add remaining
    if current_chunk.strip():
        chunk_num += 1
        chunk_doc = LegalDocument(
            id=f"{doc.id}_chunk{chunk_num}",
            title=f"{doc.title} (Part {chunk_num})",
            content=current_chunk.strip(),
            source=doc.source,
            case_type=doc.case_type,
            year=doc.year,
            jurisdiction=doc.jurisdiction,
            court=doc.court,
            citation=doc.citation,
            topic=doc.topic,
            url=doc.url,
            chunk_index=chunk_num,
        )
        chunks.append(chunk_doc)
    
    return chunks if chunks else [doc]


# ---------------------------------------------------------------------------
# RAGFlow-inspired metadata helpers (boost_terms + parent_id)
# ---------------------------------------------------------------------------

_CHUNK_SUFFIX_RE = re.compile(r"_chunk\d+$")


def derive_parent_id(doc_id: str) -> str:
    """Return the base document id by stripping a trailing '_chunkN' suffix."""
    return _CHUNK_SUFFIX_RE.sub("", doc_id)


def compute_boost_terms(content: str) -> List[str]:
    """Extract legal boost terms for a chunk; [] when disabled or on failure."""
    if not KEYWORD_BOOST_ENABLED:
        return []
    try:
        from rag_pipeline.keyword_booster import extract_boost_terms
        return extract_boost_terms(content) or []
    except Exception as e:
        logger.warning(f"Boost term extraction failed: {e}")
        return []


def store_parent_content(doc: LegalDocument, seen_parents: set) -> None:
    """Store full parent content once per chunked doc when PARENT_CHILD_ENABLED."""
    if not PARENT_CHILD_ENABLED:
        return
    if not _CHUNK_SUFFIX_RE.search(doc.id):
        return
    parent_id = derive_parent_id(doc.id)
    if parent_id in seen_parents:
        return
    seen_parents.add(parent_id)
    try:
        from rag_pipeline.parent_store import ParentStore
        ParentStore().put_parent(parent_id, doc.content)
    except Exception as e:
        logger.warning(f"ParentStore put_parent({parent_id}) failed: {e}")


# ---------------------------------------------------------------------------
# Embedding + Upsert
# ---------------------------------------------------------------------------

def generate_embedding(text: str, max_retries: int = 10, base_delay: float = 2.0, km=None) -> List[float]:
    """Generate an embedding vector using Gemini embedding API with key rotation.
    
    On 429 rate limit, rotates to the next API key instead of waiting.
    After all keys exhausted, does a cooldown wait before retrying.
    """
    import requests as http_req
    import random
    from rag_pipeline.gemini_key_manager import key_manager, search_key_manager

    # Default to the shared 12-key pool (used by the background embedder). The
    # API search/DeepSearch paths pass search_key_manager explicitly to isolate
    # them from ingestion. Fall back to the dedicated key only if the pool is empty.
    km = km or key_manager
    if not km._keys:
        km = search_key_manager

    for attempt in range(max_retries):
        # Check global cooldown
        km.check_cooldown()
        
        current_key = km.get_key()
        key_masked = km.get_key_masked()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={current_key}"
        
        try:
            response = http_req.post(url, json={
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text[:8000]}]},
            }, timeout=30)
            
            if response.status_code == 200:
                km.report_success()
                return response.json()["embedding"]["values"]
            elif response.status_code in (429, 403):
                # Rotate to next key instead of waiting. 403 = project/model
                # access denied on this key (permanent), 429 = rate limit.
                new_key = km.report_rate_limit()
                logger.warning(f"Key {key_masked} {response.status_code} (attempt {attempt+1}), rotated")
                # Small delay between key switches to be safe
                time.sleep(1)
                continue
            else:
                raise Exception(f"Embedding API error: {response.status_code} - {response.text}")
                
        except http_req.exceptions.Timeout:
            logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries}), retrying...")
            time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
            time.sleep(base_delay * (2 ** attempt))
    
    raise Exception(f"Failed to generate embedding after {max_retries} retries")


def upsert_to_pinecone(
    documents: List[LegalDocument],
    namespace: str = CHUNK_NAMESPACE,
    batch_size: int = 50,
) -> Dict[str, Any]:
    """
    Generate embeddings for documents and upsert them to Pinecone.
    Uses the pinecone Python client directly.
    """
    if not documents:
        logger.warning("No documents to upsert")
        return {"upserted": 0, "failed": 0}
    
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Also generate document summaries for multi-granularity
    doc_summaries = []
    seen_docs = set()
    for doc in documents:
        parent_id = doc.id.split("_chunk")[0]
        if parent_id not in seen_docs and doc.chunk_index == 1:
            doc_summaries.append(doc)
            seen_docs.add(parent_id)
    
    # Upsert chunks
    vectors_to_upsert = []
    failed = 0
    seen_parents = set()
    
    for i, doc in enumerate(documents):
        try:
            # Proactive delay to avoid rate limits (Gemini free tier: ~60 req/min)
            time.sleep(1.0)
            
            store_parent_content(doc, seen_parents)
            
            embedding = generate_embedding(doc.content)
            vectors_to_upsert.append({
                "id": doc.id,
                "values": embedding,
                "metadata": {
                    "title": doc.title[:200],
                    "case_name": doc.title[:200],  # For compatibility with query formatting
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
                    "boost_terms": compute_boost_terms(doc.content),
                    "parent_id": derive_parent_id(doc.id),
                },
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Embedded {i + 1}/{len(documents)}")
            
            # Batch upsert
            if len(vectors_to_upsert) >= batch_size:
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                logger.info(f"  Upserted {len(vectors_to_upsert)} vectors")
                vectors_to_upsert = []
                time.sleep(1)
        
        except Exception as e:
            failed += 1
            logger.error(f"Failed to embed/upsert {doc.id}: {e}")
    
    # Final batch
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert, namespace=namespace)
        logger.info(f"  Upserted final {len(vectors_to_upsert)} vectors")
    
    # Upsert document summaries
    if doc_summaries:
        summary_vectors = []
        for doc in doc_summaries:
            try:
                summary_text = doc.content[:1024]
                embedding = generate_embedding(summary_text)
                summary_vectors.append({
                    "id": f"doc_summary_{doc.id.split('_chunk')[0]}",
                    "values": embedding,
                    "metadata": {
                        "title": doc.title[:200],
                        "case_name": doc.title[:200],
                        "content": summary_text,
                        "source": doc.source,
                        "case_type": doc.case_type,
                        "year": doc.year,
                        "jurisdiction": doc.jurisdiction,
                        "topic": doc.topic,
                        "granularity": "document",
                    },
                })
            except Exception as e:
                logger.warning(f"Failed to embed summary for {doc.id}: {e}")
        
        if summary_vectors:
            index.upsert(vectors=summary_vectors, namespace=DOCUMENT_SUMMARY_NAMESPACE)
            logger.info(f"Upserted {len(summary_vectors)} document summaries")
    
    # Wait for consistency
    time.sleep(5)
    stats = index.describe_index_stats()
    logger.info(f"Index stats: {stats}")
    
    # Build BM25 index for hybrid search
    logger.info("Building BM25 index for hybrid search...")
    try:
        from rag_pipeline.hybrid_retriever import BM25Index
        bm25 = BM25Index()
        # Convert documents to format expected by BM25Index
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
                    "boost_terms": compute_boost_terms(doc.content),
                    "parent_id": derive_parent_id(doc.id),
                }
            })
        bm25.build(bm25_chunks)
        logger.info(f"BM25 index built with {len(bm25_chunks)} chunks")
    except Exception as e:
        logger.warning(f"BM25 index build failed (non-fatal): {e}")
    
    return {
        "upserted": len(documents) - failed,
        "failed": failed,
        "total_vectors": stats.total_vector_count,
        "namespaces": {k: v.vector_count for k, v in stats.namespaces.items()},
    }


# ---------------------------------------------------------------------------
# Source 6: CSV Dataset (employment_cases_large.csv)
# ---------------------------------------------------------------------------

def load_csv_cases(csv_path: str = None, max_cases: int = None, case_ids: list = None) -> List[LegalDocument]:
    """
    Load employment law cases from the CSV dataset.
    
    CSV columns: Caseid, URL, Case Name, Supervision/review of work, 
    Ability to hire employees, Delegation of tasks, Ownership of tools,
    Chance of profit, Risk of loss, Exclusivity of services, 
    Who sets the work hours, Where the work is performed, 
    Is the worker required to wear a uniform?, Outcome
    
    Args:
        csv_path: Path to CSV file (defaults to data/employment_cases_large.csv)
        max_cases: Maximum number of cases to load (from start)
        case_ids: Specific case IDs to load (filters during load)
    """
    import csv
    from pathlib import Path
    
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "data" / "employment_cases_large.csv"
    
    if not Path(csv_path).exists():
        logger.error(f"CSV file not found: {csv_path}")
        return []
    
    # Convert case_ids to set for O(1) lookup
    case_id_set = set(case_ids) if case_ids else None
    
    docs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            case_id = row.get('Caseid', '')
            
            # Skip if case_ids filter provided and this case not in list
            if case_id_set and case_id not in case_id_set:
                continue
            
            if max_cases and len(docs) >= max_cases:
                break
            
            # Build rich content from all features
            features = [
                f"Supervision/review of work: {row.get('Supervision/review of work', '')}",
                f"Ability to hire employees: {row.get('Ability to hire employees', '')}",
                f"Delegation of tasks: {row.get('Delegation of tasks', '')}",
                f"Ownership of tools: {row.get('Ownership of tools', '')}",
                f"Chance of profit: {row.get('Chance of profit', '')}",
                f"Risk of loss: {row.get('Risk of loss', '')}",
                f"Exclusivity of services: {row.get('Exclusivity of services', '')}",
                f"Who sets the work hours: {row.get('Who sets the work hours', '')}",
                f"Where the work is performed: {row.get('Where the work is performed', '')}",
                f"Is the worker required to wear a uniform?: {row.get('Is the worker required to wear a uniform?', '')}",
            ]
            
            content = f"""Case: {row.get('Case Name', '')}
URL: {row.get('URL', '')}
Outcome: {row.get('Outcome', '')}

Sagaz Factors Analysis:
{chr(10).join(features)}

This case was classified as: {row.get('Outcome', '')}
Based on the application of the Sagaz test factors to the working relationship."""
            
            case_id = row.get('Caseid', f'csv_{i}')
            doc_id = f"csv_{case_id}"
            
            docs.append(LegalDocument(
                id=doc_id,
                title=row.get('Case Name', f'Case {case_id}'),
                content=content,
                source="csv_dataset",
                case_type="Employment Classification Case",
                year="2024",
                jurisdiction="Ontario",
                court="ONSC",
                citation=row.get('URL', ''),
                topic="Worker Classification",
                url=row.get('URL', ''),
            ))
    
    logger.info(f"Loaded {len(docs)} cases from CSV dataset")
    return docs


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal Document Ingestion Pipeline")
    parser.add_argument("--source", choices=["canlii", "curated", "legislation", "web", "csv", "all"],
                        default="curated", help="Document source to ingest")
    parser.add_argument("--max", type=int, default=20, help="Max documents to scrape")
    parser.add_argument("--query", type=str, default="employment law", help="Search query for CanLII")
    parser.add_argument("--csv-path", type=str, default=None, help="Custom CSV file path (for csv source)")
    parser.add_argument("--upsert", action="store_true", help="Actually upsert to Pinecone")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without upserting")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("LEGAL DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 60)
    
    all_docs: List[LegalDocument] = []
    
    # Collect documents from specified sources
    if args.source in ("curated", "all"):
        docs = load_curated_documents()
        logger.info(f"Curated documents: {len(docs)}")
        all_docs.extend(docs)
    
    if args.source in ("canlii", "all"):
        docs = scrape_canlii_cases(max_cases=args.max)
        logger.info(f"CanLII scraped: {len(docs)}")
        all_docs.extend(docs)
    
    if args.source in ("web", "all"):
        docs = search_canlii_web(query=args.query, max_results=args.max)
        logger.info(f"CanLII web: {len(docs)}")
        all_docs.extend(docs)
    
    if args.source in ("legislation", "all"):
        docs = scrape_legislation()
        logger.info(f"Legislation: {len(docs)}")
        all_docs.extend(docs)
    
    if args.source in ("csv", "all"):
        docs = load_csv_cases(csv_path=args.csv_path, max_cases=args.max)
        logger.info(f"CSV dataset: {len(docs)}")
        all_docs.extend(docs)
    
    # Chunk documents
    logger.info(f"Chunking {len(all_docs)} documents...")
    chunked_docs = []
    for doc in all_docs:
        chunks = chunk_document(doc)
        chunked_docs.extend(chunks)
    logger.info(f"After chunking: {len(chunked_docs)} chunks")
    
    if args.dry_run:
        logger.info("\nDRY RUN - Documents to ingest:")
        for doc in chunked_docs[:10]:
            logger.info(f"  [{doc.source}] {doc.title} ({len(doc.content)} chars)")
        if len(chunked_docs) > 10:
            logger.info(f"  ... and {len(chunked_docs) - 10} more")
        logger.info(f"\nTotal: {len(chunked_docs)} chunks from {len(all_docs)} documents")
        return
    
    if args.upsert:
        logger.info(f"\nUpserting {len(chunked_docs)} chunks to Pinecone...")
        result = upsert_to_pinecone(chunked_docs)
        logger.info(f"Result: {result}")
    else:
        logger.info(f"\nUse --upsert to actually push to Pinecone")
        logger.info(f"Ready: {len(chunked_docs)} chunks from {len(all_docs)} documents")
    
    logger.info("DONE")


if __name__ == "__main__":
    main()