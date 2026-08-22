# ⚖️ Legal AI System — Enterprise Edition

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-AKS-326ce5.svg)](https://kubernetes.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A comprehensive, enterprise-grade legal AI platform for employment law analysis, misclassification detection, and context-aware reasoning.**

A comprehensive legal AI platform for employment law analysis, featuring:
- **RAG Pipeline**: Automated legal document collection, scaled to **500+ documents** via Pinecone
- **ML Classifier**: Random Forest model trained on **1255 cases** (100% accuracy)
- **REST API**: FastAPI service with RAG query and classification endpoints
- **Cloud Deployment**: Docker containerization with Azure Kubernetes Service deployment

### 🏭 v3.0 — ByteDance RAG Production Architecture (Latest)
Based on [ByteDance's internal RAG guideline](rag_pipeline/Byte_Dance_RAG_Guideline.md), the pipeline now implements commercial-grade patterns:
- **Hybrid Retrieval**: BM25 (Elasticsearch) + Dense Vector fusion with Reciprocal Rank Fusion and MMR diversity
- **Semantic Chunking**: Section-aware, paragraph-preserving chunking with variable sizes (128 tokens for statutes, 384 for narrative)
- **Structured Prompt Templates**: 4 legal-domain templates with intent-based auto-selection
- **Confidence Gate**: Pre-generation refuse/hedge/pass quality control
- **Multi-Layer Caching**: Embedding (1h TTL), retrieval (10m), response (5m) with LRU eviction
- **Pipeline Metrics**: Per-query structured JSONL logging with latency percentiles and cost estimation
- **Feedback Loop**: User feedback collection with root cause analysis and few-shot example export
- **Vector Store Abstraction**: Unified interface supporting Pinecone (managed) and Milvus (self-hosted, GPU-accelerated)
- **Model Optimization**: LoRA fine-tuning scaffolding, INT8 quantisation, knowledge distillation, GPU auto-scaling configs

### 🔬 v2.0 — Research-Grade Modules
- **Multi-Hop RAG**: Iterative retrieve-read-reason cycles with gap analysis and dynamic stopping
- **Knowledge Graph + Hybrid Retrieval**: Legal entity graph (NetworkX) with BFS subgraph traversal and graph-to-text linearization, fused with vector search
- **Post-Hoc Verifier**: Adversarial NLI fact-checker with claim-level citation mapping and implicit assumption validation to eliminate hallucinations
- **MCTS Legal Reasoning Agent**: Monte Carlo Tree Search over Sagaz test factors with UCB1 selection and LLM-as-judge simulation
- **Dynamic Evaluation Framework**: Anti-contamination parameterized benchmarks with difficulty-controlled test generation
- **Debiased LLM-as-Judge**: Rubric decomposition, position-swap debiasing, and length normalization for fair evaluation

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [System Architecture](#-system-architecture)
3. [Quick Start](#-quick-start)
4. [System Workflow](#-system-workflow)
5. [API Endpoints](#-api-endpoints)
6. [Model Performance](#-model-performance)
7. [File Dictionary & Project Structure](#-file-dictionary--project-structure)
8. [Verification & Testing](#-verification--testing)
9. [Maintenance & Cleanup](#-maintenance--cleanup)
10. [References & License](#-references--license)

---

## 🎯 Key Features

### Legal RAG Pipeline
- Automated collection of 500+ legal documents (Real + Synthetic)
- **Semantic chunking** with section-awareness and paragraph preservation (v3.0)
- Google Gemini embeddings (gemini-embedding-001)
- **Dual vector store**: Pinecone (managed) or Milvus (self-hosted) with HNSW tuning (v3.0)
- **Hybrid BM25 + vector retrieval** with Elasticsearch and query-type-aware weights (v3.0)
- Context-aware response generation with Gemini 2.0 Flash
- **Confidence gate**, **multi-layer caching**, and **pipeline metrics** (v3.0)

### Worker Classification Model
- Random Forest classifier trained on 1255 annotated employment cases
- 100% accuracy on test set
- Feature importance analysis for legal interpretability
- Based on Sagaz test factors (control, tools, profit/loss, integration)

### Production Infrastructure
- Docker multi-stage builds for optimized images
- Kubernetes deployment with HPA auto-scaling
- GitHub Actions CI/CD pipeline
- 99.5% uptime target with rolling updates

---

## 🏗️ System Architecture

The Legal AI platform is built on a modern, decoupled microservices architecture designed for high availability and scalable inference.

### High-Level System Flow
```mermaid
graph TD
    subgraph Data Layer
        A[CanLII Scraper] -->|PDFs| B["Document Processor<br>(SemanticChunker v3.0)"]
        C[Synthetic Generator] -->|CSV| D[ML Trainer]
    end

    subgraph Logic Layer
        B -->|Chunks| E[Gemini Embeddings]
        E -->|Vectors| F["Pinecone / Milvus"]
        B -->|Text| ES["Elasticsearch BM25"]
        D -->|Model| G[Random Forest Classifier]
        K[Knowledge Graph extractor] -->|Triples| L[NetworkX Graph DB]
        B -->|Text| K
    end

    subgraph "Application Layer (v3.0 Enhanced)"
        H[FastAPI Gateway] -->|Query| QC[Query Classifier]
        QC -->|keyword| ES
        QC -->|semantic| F
        QC -->|hybrid| BOTH["BM25 + Vector"]
        ES --> RRF["RRF Fusion + MMR"]
        F --> RRF
        BOTH --> RRF
        RRF --> PT[Prompt Template Auto-Select]
        PT --> J[Gemini 2.0 Flash]
        J --> CG[Confidence Gate]
        CG --> VER[NLI Verifier]
        H -->|Classify| G
        H -->|Reasoning| M[MCTS Agent]
        M --> QC
    end
```

### Multi-Hop RAG & MCTS Agent Workflow
To support complex legal analysis, v2.0 introduces advanced orchestration techniques:
```mermaid
sequenceDiagram
    participant User
    participant Gateway as FastAPI Gateway
    participant MCTS as MCTS Reasoning Agent
    participant RAG as Multi-Hop RAG
    participant KG as Knowledge Graph
    participant DB as Pinecone DB
    participant LLM as Gemini 2.0

    User->>Gateway: POST /classify/reasoning
    Gateway->>MCTS: Initialize Search Tree
    loop Monte Carlo Tree Search
        MCTS->>MCTS: Select Node (UCB1)
        MCTS->>MCTS: Expand Hypothesis
        MCTS->>RAG: Generate Evidence Query
        RAG->>DB: Vector Search
        RAG->>KG: Subgraph Traversal
        RAG->>LLM: Gap Analysis
        LLM-->>RAG: Missing context
        RAG-->>MCTS: Comprehensive Context
        MCTS->>LLM: Simulate Reward (Judge)
        LLM-->>MCTS: Score
        MCTS->>MCTS: Backpropagate Score
    end
    MCTS-->>Gateway: Best Reasoning Trace
    Gateway-->>User: JSON Response
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- API Keys: Gemini (Google AI Studio), Pinecone
- **v3.0 optional**: Elasticsearch 8.x, Milvus 2.4+ (for production BM25 and self-hosted vector store)

### Installation

```bash
# Clone repository
git clone https://github.com/johnnietse/legal-ai.git
cd legal-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### v3.0 Environment Variables
```bash
# Elasticsearch (production BM25)
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_API_KEY=your_key

# Milvus (optional, self-hosted vector store)
MILVUS_HOST=localhost
MILVUS_PORT=19530
VECTOR_STORE_BACKEND=pinecone  # or "milvus"

# BM25 backend
BM25_BACKEND=elasticsearch     # or "local" for development

# HNSW tuning preset
HNSW_PRESET=production         # development | production | high_recall | billion_scale
```

### Running Locally

```bash
# Start API server
python -m api.main

# Or use Docker Compose
docker-compose up
```

Access API docs at: http://localhost:8000/docs

---

## 🔄 System Workflow

The system integrates three main pipelines:

1.  **Data Ingestion & Scaling**:
    *   Synthetic generation of 1200+ cases -> `data/employment_cases_large.csv`.
    *   Manual/Synthetic generation of 500+ legal documents -> Pinecone Index.
2.  **Machine Learning Training**:
    *   `train_model.py` loads CSV -> trains Random Forest -> saves `models/worker_classifier.joblib`.
3.  **RAG Knowledge Retrieval**:
    *   `api/main.py` -> `LegalRAGPipeline` -> Query Pinecone for semantic matches -> Gemini for answer generation.

---

## 🔌 API Endpoints

The system provides a comprehensive REST API powered by FastAPI, featuring interactive Swagger documentation (`/docs`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/rag/query` | POST | Single-hop legal knowledge query (v3.0: hybrid retrieval + prompt templates) |
| `/rag/query/multi-hop` | POST | ⭐ Multi-hop iterative retrieval for complex questions |
| `/rag/query/smart` | POST | ⭐ Auto-routes between single/multi-hop based on complexity |
| `/rag/verify` | POST | ⭐ Standalone fact-checking and assumption validation |
| `/rag/stats` | GET | 🏭 Pipeline statistics: latency, cost, cache hit rates, quality scores |
| `/api/feedback` | POST | 🏭 Submit user feedback (useful/not_useful/wrong) with error categorisation |
| `/api/feedback/summary` | GET | 🏭 Feedback analysis: ratings, flagged queries, root cause breakdown |
| `/classify` | POST | Worker classification (Random Forest) |
| `/classify/reasoning` | POST | ⭐ MCTS-based classification with full reasoning trace |
| `/evaluate/generate-suite` | POST | ⭐ Generate anti-contamination evaluation suite |
| `/evaluate/judge` | POST | ⭐ Score responses with debiased LLM judge |

---

## 📊 Model Performance (Scaled)

| Metric | Value |
|--------|-------|
| **Accuracy** | **100%** |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |
| Cross-Validation | 99.8% |
| Training Data | 1255 Cases |

### Top Classification Factors
1. **Did worker require uniform?** (0.255)
2. **Delegation of tasks** (0.165)
3. **Ownership of tools** (0.155)
4. **Exclusivity** (0.145)

---

## 📂 File Dictionary & Project Structure

### **Core Components**
| File | Role & Description |
|------|-------------------|
| **`config.py`** | **Central Configuration**: Manages API keys (Gemini, Pinecone), file paths, and system constants. Single source of truth. |
| **`requirements.txt`** | **Dependencies**: Lists all Python libraries required (FastAPI, Pinecone, Pandas, scikit-learn). |

### **RAG Pipeline (`rag_pipeline/`)**
| File | Role & Description |
|------|-------------------|
| `pipeline.py` | **Orchestrator**: Ties together scraping, processing, embedding, and querying. The main entry point for RAG operations. |
| `canlii_scraper.py` | **Data Ingestion**: Handles web scraping from CanLII, including rate limiting and CAPTCHA detection. |
| `document_processor.py` | **Preprocessing**: Extracts text from PDFs and chunks it intelligently (preserving legal context). |
| `embeddings.py` | **Vectorization**: Interface to Google's Gemini Embedding API (gemini-embedding-001). |
| `pinecone_client.py` | **Storage**: Manages connection to Pinecone Vector DB, including upserting and semantic search. |
| `rag_query.py` | **Retrieval**: Formulates prompts and calls Gemini for answer generation based on retrieved context. |

### **Advanced RAG Modules (`rag_pipeline/` — v2.0)**
| File | Role & Description |
|------|-------------------|
| `multi_hop_retriever.py` | **Multi-Hop RAG**: State-machine iterative retrieval with gap analysis, query reformulation, and dynamic stopping criteria. Research contribution: the `_should_stop` function implements retrieval trajectory control. |
| `knowledge_graph.py` | **Legal Knowledge Graph**: LLM-based entity/relation extraction, NetworkX directed graph, BFS subgraph traversal, and graph-to-text linearization for LLM consumption. |
| `graph_retriever.py` | **Hybrid Retriever**: Fuses Pinecone vector search with knowledge graph traversal. Merges semantic similarity results with structured relationship context. |
| `verifier.py` | **Fact-Checker**: Uses adversarial NLI to extract implicit assumptions, map citations, and auto-correct hallucinations. |
| `legal_reasoning_agent.py` | **MCTS Agent**: Monte Carlo Tree Search over Sagaz classification factors with UCB1 selection, LLM-as-judge simulation, and backpropagation. Implements inference-time compute scaling. |

### **ByteDance RAG Enhancements (`rag_pipeline/` — v3.0)** 🏭
| File | Role & Description |
|------|-------------------|
| `hybrid_retriever.py` | **Hybrid Search**: BM25 keyword + dense vector fusion with Reciprocal Rank Fusion (RRF), query-type classifier (keyword/semantic/hybrid), and Maximum Marginal Relevance (MMR) diversity reranking. §5.2 |
| `prompt_templates.py` | **Prompt Library**: 4 domain-specific templates (worker classification, notice period, risk analysis, general) with intent-based auto-selector. §6.2 |
| `confidence_gate.py` | **Quality Gate**: Pre-generation confidence checking — refuses low-confidence, hedges borderline, passes high-confidence answers. §6.3.1 |
| `query_cache.py` | **Multi-Layer Cache**: 3-tier TTL cache (embedding 1h, retrieval 10m, response 5m) with LRU eviction and disk persistence. §6.4.2 |
| `metrics.py` | **Pipeline Metrics**: Per-query structured JSONL logging with timing, cost estimation, latency percentiles (p50/p95), and weekly summaries. §8.1 |
| `feedback_analyzer.py` | **Feedback Loop**: JSONL-backed user feedback (useful/not_useful/wrong), root cause analysis by system layer, flagged query detection, few-shot example export. §6.3.3 |
| `vector_store.py` | **Vector Store Abstraction**: Unified interface for Pinecone (managed) and Milvus (self-hosted, GPU-accelerated). HNSW tuning presets from development to billion-scale. §4.2.1 |
| `search_engine.py` | **Elasticsearch BM25**: Production sparse retrieval with legal-domain custom analyser, legal synonym expansion, and metadata-filtered search. §5.2 |
| `model_optimization.py` | **Model Optimization**: LoRA fine-tuning (PEFT), INT8/4-bit quantisation (BitsAndBytes), teacher-student distillation, GPU auto-scaling profiles, cross-region deployment configs. §7–§8 |

### **Evaluation Framework (`evaluation/` — v2.0)**
| File | Role & Description |
|------|-------------------|
| `dynamic_benchmark.py` | **Anti-Contamination Benchmarks**: Parameterized test case generation with randomized entities, difficulty-controlled factor distributions, and 6-dimensional evaluation rubrics. |
| `llm_judge.py` | **Debiased Judge**: LLM-as-Judge with rubric decomposition (anti-self-enhancement), position swapping (anti-position-bias), and length normalization (anti-length-bias). |
| `bias_detector.py` | **Bias Auditor**: Automated detection and quantification of position, length, and self-enhancement biases in judge models. |
| `benchmark_runner.py` | **Benchmark CLI**: End-to-end evaluation orchestration with per-difficulty breakdowns and multi-dimensional scoring reports. |

### **ML Classifier (`ml_classifier/`)**
| File | Role & Description |
|------|-------------------|
| `train_classifier.py` | **Training Logic**: Loads CSV data, trains Random Forest model, and performs hyperparameter tuning. |
| `model_inference.py` | **Prediction**: Helper for making predictions with the trained `worker_classifier.joblib` model. |

### **API & Deployment**
| File | Role & Description |
|------|-------------------|
| `api/main.py` | **Gateway**: FastAPI application defining REST endpoints (`/rag/query`, `/classify`). |
| `k8s/` | **Orchestration**: Kubernetes manifests (Deployment, Service) for cloud scaling. |
| `Dockerfile` | **Containerization**: Multistage build instructions for creating the production image. |

### **Data Management (`data/`)**
| File | Role & Description |
|------|-------------------|
| `generate_large_dataset.py` | **Scale Generator**: Creates thousands of synthetic employment cases for robust training. |
| `employment_cases_large.csv` | **Training Data**: The gold-standard dataset (1255 cases) used for the ML model. |

### Directory Tree

```text
Law_AI/
├── rag_pipeline/                  # RAG pipeline components
│   ├── canlii_scraper.py          # CanLII web scraper
│   ├── document_processor.py      # PDF extraction, chunking + SemanticChunker (v3)
│   ├── embeddings.py              # Gemini embeddings
│   ├── pinecone_client.py         # Vector database client
│   ├── rag_query.py               # Query interface (v3: full ByteDance pipeline)
│   ├── pipeline.py                # Main orchestrator
│   ├── multi_hop_retriever.py     # ⭐ Multi-hop iterative retrieval
│   ├── knowledge_graph.py         # ⭐ Legal entity knowledge graph
│   ├── graph_retriever.py         # ⭐ Hybrid vector+graph retriever
│   ├── legal_reasoning_agent.py   # ⭐ MCTS classification agent
│   ├── verifier.py                # ⭐ NLI fact-checker
│   ├── hybrid_retriever.py        # 🏭 BM25+vector fusion, RRF, MMR
│   ├── prompt_templates.py        # 🏭 Domain prompt templates
│   ├── confidence_gate.py         # 🏭 Pre-generation quality gate
│   ├── query_cache.py             # 🏭 Multi-layer TTL cache
│   ├── metrics.py                 # 🏭 Pipeline metrics & logging
│   ├── feedback_analyzer.py       # 🏭 User feedback loop
│   ├── vector_store.py            # 🏭 Pinecone/Milvus abstraction
│   ├── search_engine.py           # 🏭 Elasticsearch BM25
│   └── model_optimization.py      # 🏭 LoRA, quantisation, deployment
├── evaluation/                    # ⭐ Evaluation framework
│   ├── dynamic_benchmark.py       # Anti-contamination test generation
│   ├── llm_judge.py               # Debiased LLM-as-Judge
│   ├── bias_detector.py           # Bias detection & auditing
│   └── benchmark_runner.py        # CLI benchmark orchestrator
├── ml_classifier/                 # ML classification
│   ├── train_classifier.py        # Random Forest training
│   └── model_inference.py         # Inference API
├── api/                           # FastAPI service
│   └── main.py                    # REST endpoints (v1 + v2 + v3)
├── data/                          # Data storage
│   ├── employment_cases_large.csv
│   ├── cache/                     # 🏭 Query cache (auto-generated)
│   └── generate_large_dataset.py
├── scripts/                       # 🏭 Deployment & execution scripts
│   ├── reindex_local.py           # Local semantic re-indexing pipeline
│   ├── bootstrap_elasticsearch.py # Elasticsearch index creation & loading
│   └── lora_finetune.py           # LoRA training & distillation pipeline
├── tests/                         # Test suites (85+ total tests)
│   ├── test_advanced_modules.py   # Tests for v2 modules
│   ├── test_v3_bytedance.py       # Tests for v3 ByteDance modules
│   ├── test_system_integration.py # E2E system integration tests
│   └── test_main.py               # API and core functionality tests
├── logs/metrics/                  # 🏭 Structured metrics JSONL
├── k8s/                           # Kubernetes manifests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── config.py                      # Central config (60+ v3 constants)
```

---

## ✅ Verification & Testing

To verify the entire system integration (Data, ML, and RAG connection), run the included test suite:

```bash
# Run the complete test suite (v1, v2, and v3 modules, ~85 tests)
python -m pytest tests/ -v --tb=short

# Run specifically the v3 ByteDance RAG tests
python -m pytest tests/test_v3_bytedance.py -v
```

Expected output for integration:
> ✅ Data Verified
> ✅ ML Pipeline Verified
> ✅ RAG Pipeline Verified
> SYSTEM INTEGRATION TEST PASSED

### Individual Component Testing
```bash
# Test RAG scraping/processing
python rag_pipeline/pipeline.py --query "What is the Sagaz test?"

# Test ML Prediction interaction
python -m ml_classifier.train_classifier --demo
```

---

## 🛠️ Maintenance

### Scaling Up
```bash
# Generate more training data
python data/generate_large_dataset.py

# Populate RAG (takes ~10m for 500 docs)
python populate_pinecone_large.py
```

### Verification
```bash
python verify_pinecone.py
python train_model.py
```

---

## 🧹 Cleanup Recommendations

The following files and directories are **safe to remove** as they are not required for the core system:

| Path | Reason |
|------|--------|
| `openjustice-rag/` | Empty reference project, unused |
| `create_landmark_cases.py` | One-time script for generating test cases |
| `fetch_real_cases.py` | Experimental CanLII URL fetcher |
| `real_cases_to_scrape.csv` | Sample output from above script |
| `.pytest_cache/` | Test cache, auto-generated |
| `__pycache__/` | Python bytecode cache |
| `logs/` | Runtime logs (if empty) |

**To remove all at once:**
```powershell
Remove-Item -Recurse -Force openjustice-rag, .pytest_cache, __pycache__, logs
Remove-Item create_landmark_cases.py, fetch_real_cases.py, real_cases_to_scrape.csv
```

---

## 📚 References & License

- [Sagaz Industries Test](https://scc-csc.lexum.com/scc-csc/scc-csc/en/item/1870/index.do) - Supreme Court worker classification test
- [CanLII](https://www.canlii.org) - Canadian Legal Information Institute
- [Deel Lab for Global Employment](https://www.deel.com/research) - Research initiative

This system is designed for the LLM x Law Hackathon, addressing challenges in:
- Legal document retrieval and analysis
- Worker misclassification detection
- AI-assisted legal research

### License

MIT License - See [LICENSE](LICENSE) file for details.
