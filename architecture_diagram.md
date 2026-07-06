# Deel Lab Legal AI Architecture Diagram

Below is the scientific research grade, Apple symposium-level architecture diagram for the Deel Lab Legal AI system. It captures the complex integration of your Python RAG pipeline, the Random Forest Machine Learning classifier, and the highly-available containerized microservices running on Azure Kubernetes Service.

```mermaid
flowchart TB
    %% Apple-Symposium Style Definitions
    classDef entrypoint fill:#1D1D1F,stroke:#1D1D1F,color:#F5F5F7,rx:8px,ry:8px,font-family:-apple-system,font-weight:bold;
    classDef compute fill:#FFFFFF,stroke:#D2D2D7,stroke-width:2px,color:#1D1D1F,rx:8px,ry:8px,font-family:-apple-system;
    classDef data fill:#F5F5F7,stroke:#86868B,stroke-width:1px,color:#1D1D1F,rx:8px,ry:8px,font-family:-apple-system;
    classDef cloud fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1D1D1F,rx:12px,ry:12px,font-family:-apple-system,font-weight:600;
    classDef model fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#1D1D1F,rx:8px,ry:8px,font-family:-apple-system;

    %% External Interfaces
    CLIENT["Client Applications"]:::entrypoint

    %% Cloud Infrastructure Subsystem
    subgraph AKS [☁️ Azure Kubernetes Service Cluster]
        direction TB
        
        GATEWAY["FastAPI REST API Gateway<br><i>High-Availability Pods (99.5% Uptime)</i>"]:::compute
        
        %% RAG Pipeline Subsystem (v3.0 ByteDance Architecture)
        subgraph RAG [RAG Services Pipeline]
            direction TB
            SCRAPER["Data Ingestion Engine<br><i>Selenium & BS4 (10K+ Cases)</i>"]:::data
            CHUNKER["Semantic Document Processor<br><i>Section-Aware Chunking (v3.0)</i>"]:::compute
            EMBEDDING["Embedding Model<br><i>Gemini gemini-embedding-001</i>"]:::model
            
            BM25_DB[("Elasticsearch<br><i>Sparse BM25 Index</i>")]:::data
            VECTOR_DB[("Pinecone / Milvus<br><i>Dense Semantic Index</i>")]:::data
            
            CACHE["Multi-Layer Cache<br><i>TTL-based Response Caching</i>"]:::data
            Q_CLASS["Query Classifier<br><i>Semantic / Keyword / Hybrid</i>"]:::compute
            FUSION["RRF & MMR Fusion<br><i>Diversity Re-ranking</i>"]:::compute
            TEMPLATES["Prompt Auto-Selector<br><i>Domain-Specific Templates</i>"]:::compute
            GENERATOR["LLM Generation Engine<br><i>Gemini 2.0 Flash</i>"]:::model
            CONFIDENCE["Confidence Gate<br><i>Pass / Hedge / Refuse</i>"]:::compute
            
            SCRAPER -->|Raw Case Law| CHUNKER
            CHUNKER -->|Semantic Chunks| EMBEDDING
            CHUNKER -->|Raw Text| BM25_DB
            EMBEDDING -->|Vectors| VECTOR_DB
            
            Q_CLASS -->|Keyword| BM25_DB
            Q_CLASS -->|Semantic| VECTOR_DB
            Q_CLASS -->|Hybrid| BM25_DB
            Q_CLASS -->|Hybrid| VECTOR_DB
            
            BM25_DB --> FUSION
            VECTOR_DB --> FUSION
            FUSION --> TEMPLATES
            TEMPLATES --> GENERATOR
            GENERATOR --> CONFIDENCE
        end
        
        %% Machine Learning Subsystem
        subgraph ML [Machine Learning Classification Pipeline]
            direction TB
            DS["Employment Law Dataset<br><i>1260+ Annotated Cases</i>"]:::data
            TRAINING["Hyperparameter Tuning<br><i>GridSearchCV</i>"]:::compute
            RF_MODEL["Random Forest Classifier<br><i>Predicts Worker Classification</i>"]:::model
            INTERP["Model Interpretability Engine<br><i>Gini Importance</i>"]:::compute
            LORA["LoRA Fine-Tuning Pipeline<br><i>Feedback-driven (v3.0)</i>"]:::compute
            
            DS --> TRAINING
            TRAINING --> RF_MODEL
            RF_MODEL --> INTERP
            LORA -.-> GENERATOR
        end
        
        GATEWAY --> CACHE
        CACHE -->|Miss| Q_CLASS
        CONFIDENCE --> GATEWAY
        GATEWAY -->|POST /classify| RF_MODEL
    end

    %% DevOps Subsystem
    subgraph CICD [DevOps & Delivery Pipeline]
        direction LR
        GHA["GitHub Actions CI/CD<br><i>Automated Testing (+80% Release Cycles)</i>"]:::compute
        DOCKER["Multi-Stage Docker Builds<br><i>Footprint Optimized Images</i>"]:::compute
        ACR["Azure Container Registry<br><i>Artifact Storage</i>"]:::data
        
        GHA --> DOCKER
        DOCKER --> ACR
    end

    %% Global Routing
    CLIENT -->|API Requests| GATEWAY
    ACR -.->|Liveness/Readiness Automated Deployments| AKS

    %% Modern Apple-esque Subgraph Styles
    style AKS fill:#FFFFFF,stroke:#D2D2D7,stroke-width:2px,rx:16px,ry:16px
    style RAG fill:#FBFBFD,stroke:#E5E5EA,stroke-width:1px,rx:12px,ry:12px
    style ML fill:#FBFBFD,stroke:#E5E5EA,stroke-width:1px,rx:12px,ry:12px
    style CICD fill:#F5F5F7,stroke:#D2D2D7,stroke-width:1px,rx:12px,ry:12px
```

## Highlights of this Architecture Diagram

1. **Clean Segregation of Concerns**: The system is neatly divided into DevOps (CI/CD), API Layer (FastAPI), RAG Services, and ML Pipelines, directly reflecting your Dockerized and Kubernetes-orchestrated architecture.
2. **ByteDance v3.0 Enhancements (New)**: The RAG pipeline now mirrors enterprise standards, featuring:
   - *Hybrid Search* combining Elasticsearch (BM25) and Pinecone/Milvus (Dense Vectors) with RRF/MMR fusion.
   - *Quality Gates* using Confidence Gating (Pass/Hedge/Refuse) to prevent hallucinations.
   - *Model Optimization* with LoRA fine-tuning and distillation pipelines.
3. **Impact-Oriented Annotations**: Key achievements are embedded directly in the visual nodes:
   - *10K+ CanLII Cases* on the Web Scraper.
   - *1260+ Annotated Cases* on the ML side.
   - *99.5% Uptime* and *+80% Release Cycles* on the Kubernetes and CI/CD elements.
4. **Apple Symposium Aesthetics**: Employs the `San Francisco` (`-apple-system`) font stack, muted elegant color palettes (cool grays for infrastructure, soft blues for data, muted reds for AI/ML processes), and rounded, border-less designs (`rx: 16px`) for a premium presentation.
