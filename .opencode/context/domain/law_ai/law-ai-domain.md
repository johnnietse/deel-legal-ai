# Law AI Domain Knowledge

## Project Purpose
Legal AI RAG (Retrieval-Augmented Generation) system for Canadian employment law. Provides AI-powered legal research and document analysis with:

- **Semantic search** over legal case law (10K+ CanLII cases)
- **BM25 sparse retrieval** for keyword matching with legal synonym expansion
- **Hybrid retrieval** combining vector + sparse search
- **Multi-hop reasoning** for complex legal queries
- **Confidence scoring** and answer verification

## Key Components

### Data Pipeline
- `data/semantic_index/chunks.jsonl` - 73 indexed legal chunks
- `data/feedback.jsonl` - 1,016 feedback entries for LoRA fine-tuning
- ES index `deel-legal-chunks` with legal synonym filter

### RAG Pipeline (`rag_pipeline/`)
| Module | Function | Backend |
|--------|----------|---------|
| `vector_store.py` | Vector embeddings storage | Milvus (self-hosted) / Pinecone (cloud) |
| `search_engine.py` | Sparse BM25 search | Elasticsearch 8.12 / local |
| `embeddings.py` | Dense embedding generation | Gemini gemini-embedding-001 |
| `hybrid_retriever.py` | Combined search | RRF fusion + MMR diversity |

### API (`api/main.py`)
- FastAPI application on port 8000
- Restricted CORS: `deel.ai` + `localhost:3000`
- Input validation: 10-5000 chars
- Endpoints: `/health`, `/query`, `/feedback`

### Infrastructure
| Service | Tech | Port |
|---------|------|------|
| API | FastAPI + Uvicorn (4 workers) | 8000 |
| ES | Elasticsearch 8.12 (single-node) | 9200 |
| Milvus | Milvus 2.3.5 standalone | 19530 |
| Milvus Meta | etcd v3.5.5 | 2379 |
| Milvus Storage | MinIO | 9000 |

### ML Training
- LoRA fine-tuning (QLoRA with bitsandbytes)
- Base model: Llama-3-8B-Instruct
- Training data: 500+ feedback entries from `data/feedback.jsonl`
- Script: `scripts/lora_finetune.py`

## API Keys
- **Gemini**: Revoked (needs replacement at aistudio.google.com)
- **Pinecone**: Revoked (needs replacement)

## Key Decisions
- MilvusClient > pymilvus ORM (future-proof, simpler API)
- `gemini-embedding-001` > `text-embedding-004` (current API)
- Non-root Docker user (UID 1000) > root (security)
- HPA config in `hpa-gpu.yaml` > inline in `service.yaml` (modularity)
