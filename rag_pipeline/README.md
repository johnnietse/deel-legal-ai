# Legal RAG Pipeline

This module handles the **Retrieval Augmented Generation (RAG)** component of the system.

## 📂 Components

| File | Purpose |
|------|---------|
| `pipeline.py` | **Entry Point**: Orchestrates scraping -> processing -> embedding -> storage. |
| `canlii_scraper.py` | **Scraper**: Fetches legal cases from CanLII (Selenium/BS4). |
| `document_processor.py` | **Processor**: Cleans text and splits into context-aware chunks (512 tokens). |
| `embeddings.py` | **Embedder**: Generates vector embeddings using `text-embedding-004`. |
| `pinecone_client.py` | **Database**: Manages Pinecone index `deel-legal-cases`. |
| `rag_query.py` | **Query Engine**: Retrieves documents and generates answers via Gemini 2.0 Flash. |

## 🚀 Usage

### 1. Verification
Run a test query to verify connection:
```bash
python rag_pipeline/rag_query.py
```

### 2. Full Pipeline execution
To scrape and ingest new data:
```python
from rag_pipeline.pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()
pipeline.run_full_pipeline(max_cases=10)
```
