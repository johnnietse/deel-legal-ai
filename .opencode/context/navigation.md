# Law AI Deel — Context Navigation

> Top-level map of all context files for the Legal AI RAG System.

## Project Overview

Legal AI RAG System for employment law analysis. Core capabilities:
- **RAG Pipeline**: CanLII scraping → semantic chunking → embedding (Gemini) → dual-store (Pinecone + Elasticsearch BM25) → hybrid retrieval with RRF fusion
- **ML Classifier**: Random Forest on 10 Sagaz-test factors, trained on 1255 cases
- **API**: FastAPI with RAG query, classification, MCTS reasoning, evaluation, and feedback endpoints
- **Deployment**: Docker multi-stage, Kubernetes (AKS) with HPA, cross-region via Karmada

## Context Map

### Core Standards
| File | Purpose |
|------|---------|
| [Core/Standards/Code Quality](core/standards/code-quality.md) | Python coding standards for this project (imports, logging, retry patterns, config) |
| [Core/Standards/Testing](core/standards/testing.md) | Testing patterns (pytest fixtures, mocking strategy, what to test) |

### Core Workflows
| File | Purpose |
|------|---------|
| [Core/Workflows/Development](core/workflows/development.md) | End-to-end dev workflow: discover → propose → approve → implement |

### Domain Knowledge
| File | Purpose |
|------|---------|
| [Domain/Law AI/Navigation](domain/law_ai/navigation.md) | Domain-specific context files map |
| [Domain/Law AI/Legal RAG](domain/law_ai/legal-rag.md) | Legal RAG architecture, ByteDance patterns, multi-hop, MCTS |

### Processes
| File | Purpose |
|------|---------|
| [Processes/Pipeline](processes/pipeline.md) | RAG pipeline ingestion and query flow |
| [Processes/Deployment](processes/deployment.md) | Docker and Kubernetes deployment process |

### Templates
| File | Purpose |
|------|---------|
| [Templates/PR](templates/pr.md) | Pull request template for this project |

## Quick Reference

### Key Files (source)
| Path | Role |
|------|------|
| `rag_pipeline/pipeline.py` | Pipeline orchestrator (scrape → process → embed → upsert → query) |
| `rag_pipeline/hybrid_retriever.py` | BM25Index, QueryClassifier, RRF, MMR, HybridRetriever |
| `rag_pipeline/rag_query.py` | Query interface with cache, confidence gate, metrics |
| `rag_pipeline/vector_store.py` | Abstract vector store (Pinecone/Milvus/Local) + HNSW presets |
| `rag_pipeline/embeddings.py` | GeminiEmbeddings + GeminiChat |
| `rag_pipeline/confidence_gate.py` | Post-generation hallucination guard |
| `api/main.py` | FastAPI app with all endpoints |
| `ml_classifier/train_classifier.py` | Random Forest training + prediction |
| `config.py` | All configuration (RAG, ML, K8s, ByteDance params) |
| `scripts/lora_finetune.py` | LoRA fine-tuning cycle |
| `tests/test_v3_bytedance.py` | Comprehensive ByteDance module tests |
| `tests/test_system_integration.py` | Integration tests across components |

### Key Patterns

- **Lazy imports**: `rag_pipeline/__init__.py` uses `__getattr__` for lazy loading
- **Factory pattern**: `create_vector_store()`, `create_bm25_engine()` in vector_store and search_engine
- **Dataclass responses**: `HybridResult`, `ConfidenceReport`, `RAGResponse`, `VectorRecord`
- **Retry**: `@retry` from tenacity on all Gemini API calls
- **Config-driven**: All tunable parameters in `config.py` (ByteDance sections)
- **Namespace pattern**: Vector stores use `namespace` parameter for multi-tenant separation

## File Size Limits

| Category | Max Lines |
|----------|-----------|
| Concepts | <100 |
| Guides | <150 |
| Examples | <80 |
| Lookup | <100 |
| Errors | <150 |
