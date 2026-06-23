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
        
        %% RAG Pipeline Subsystem
        subgraph RAG [RAG Services Pipeline]
            direction TB
            SCRAPER["Data Ingestion Engine<br><i>Selenium & BS4 (10K+ CanLII Cases)</i>"]:::data
            CHUNKER["Semantic Document Processor<br><i>Structure-Aware Chunking (-30% Hallucinations)</i>"]:::compute
            EMBEDDING["Embedding Model<br><i>Gemini text-embedding-004</i>"]:::model
            VECTOR_DB[("Pinecone Vector Database<br><i>Scalable Semantic Indexing</i>")]:::data
            GENERATOR["LLM Generation Engine<br><i>Gemini 2.0 Flash</i>"]:::model
            
            SCRAPER -->|Raw Case Law| CHUNKER
            CHUNKER -->|Overlapped Semantic Chunks| EMBEDDING
            EMBEDDING -->|Vector Embeddings| VECTOR_DB
            VECTOR_DB -->|Context Retrieval| GENERATOR
        end
        
        %% Machine Learning Subsystem
        subgraph ML [Machine Learning Classification Pipeline]
            direction TB
            DS["Employment Law Dataset<br><i>1260+ Annotated Cases</i>"]:::data
            TRAINING["Hyperparameter Tuning<br><i>GridSearchCV</i>"]:::compute
            RF_MODEL["Random Forest Classifier<br><i>Predicts Worker Classification</i>"]:::model
            INTERP["Model Interpretability Engine<br><i>Gini Importance across 10 Sagaz Factors</i>"]:::compute
            
            DS --> TRAINING
            TRAINING --> RF_MODEL
            RF_MODEL --> INTERP
        end
        
        GATEWAY -->|POST /rag/query| GENERATOR
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
2. **Impact-Oriented Annotations**: Key achievements are embedded directly in the visual nodes:
   - *10K+ CanLII Cases* on the Web Scraper.
   - *-30% Hallucinations* through Structure-Aware Chunking.
   - *1260+ Annotated Cases* and *10 Sagaz Factors* on the ML side.
   - *99.5% Uptime* and *+80% Release Cycles* on the Kubernetes and CI/CD elements.
3. **Apple Symposium Aesthetics**: Employs the `San Francisco` (`-apple-system`) font stack, muted elegant color palettes (cool grays for infrastructure, soft blues for data, muted reds for AI/ML processes), and rounded, border-less designs (`rx: 16px`) for a premium presentation.
