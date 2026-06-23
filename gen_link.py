import base64
import json

code = """
flowchart TB
    classDef entrypoint fill:#1D1D1F,stroke:#1D1D1F,color:#F5F5F7,rx:8px,ry:8px,font-family:-apple-system,font-weight:bold;
    classDef compute fill:#FFFFFF,stroke:#D2D2D7,stroke-width:2px,color:#1D1D1F,rx:8px,ry:8px,font-family:-apple-system;
    classDef data fill:#F5F5F7,stroke:#86868B,stroke-width:1px,color:#1D1D1F,rx:8px,ry:8px,font-family:-apple-system;
    classDef cloud fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1D1D1F,rx:12px,ry:12px,font-family:-apple-system,font-weight:600;
    classDef model fill:#FCE8E6,stroke:#EA4335,stroke-width:2px,color:#1D1D1F,rx:8px,ry:8px,font-family:-apple-system;

    CLIENT["💻 Client Applications"]:::entrypoint

    subgraph AKS [☁️ Azure Kubernetes Service Cluster]
        direction TB
        GATEWAY["⚡ FastAPI REST Gateway<br><i>High-Availability Pods (99.5% Uptime)</i>"]:::compute
        
        subgraph RAG [🔍 RAG Services Pipeline]
            direction TB
            SCRAPER["🌐 Data Ingestion Engine<br><i>Selenium & BS4 (10K+ CanLII Cases)</i>"]:::data
            CHUNKER["✂️ Semantic Document Processor<br><i>Structure-Aware Chunking (-30% Hallucinations)</i>"]:::compute
            EMBEDDING["🧬 Embedding Model<br><i>Gemini text-embedding-004</i>"]:::model
            VECTOR_DB[("🗄️ Pinecone Vector Database<br><i>Scalable Semantic Indexing</i>")]:::data
            GENERATOR["🧠 LLM Generation Engine<br><i>Gemini 2.0 Flash</i>"]:::model
            
            SCRAPER -->|Raw Case Law| CHUNKER
            CHUNKER -->|Overlapped Semantic Chunks| EMBEDDING
            EMBEDDING -->|Vector Embeddings| VECTOR_DB
            VECTOR_DB -->|Context Retrieval| GENERATOR
        end
        
        subgraph ML [📊 ML Classification Pipeline]
            direction TB
            DS["📄 Employment Law Dataset<br><i>1260+ Annotated Cases</i>"]:::data
            TRAINING["⚙️ Hyperparameter Tuning<br><i>GridSearchCV</i>"]:::compute
            RF_MODEL["🌳 Random Forest Classifier<br><i>Predicts Worker Classification</i>"]:::model
            INTERP["⚖️ Model Interpretability Engine<br><i>Gini Importance across 10 Sagaz Factors</i>"]:::compute
            
            DS --> TRAINING
            TRAINING --> RF_MODEL
            RF_MODEL --> INTERP
        end
        
        GATEWAY -->|POST /rag/query| GENERATOR
        GATEWAY -->|POST /classify| RF_MODEL
    end

    subgraph CICD [🚀 DevOps & Delivery Pipeline]
        direction LR
        GHA["🐙 GitHub Actions CI/CD<br><i>Automated Testing (+80% Release Cycles)</i>"]:::compute
        DOCKER["🐳 Multi-Stage Docker Builds<br><i>Footprint Optimized Images</i>"]:::compute
        ACR["📦 Azure Container Registry<br><i>Artifact Storage</i>"]:::data
        
        GHA --> DOCKER
        DOCKER --> ACR
    end

    CLIENT -->|API Requests| GATEWAY
    ACR -.->|Liveness/Readiness Deployments| AKS

    style AKS fill:#FFFFFF,stroke:#D2D2D7,stroke-width:2px,rx:16px,ry:16px
    style RAG fill:#FBFBFD,stroke:#E5E5EA,stroke-width:1px,rx:12px,ry:12px
    style ML fill:#FBFBFD,stroke:#E5E5EA,stroke-width:1px,rx:12px,ry:12px
    style CICD fill:#F5F5F7,stroke:#D2D2D7,stroke-width:1px,rx:12px,ry:12px
"""

payload = {
    "code": code,
    "mermaid": "{\n  \"theme\": \"default\"\n}",
    "autoSync": True,
    "updateDiagram": True
}

json_payload = json.dumps(payload)
base64_str = base64.urlsafe_b64encode(json_payload.encode('utf-8')).decode('utf-8')
print("https://mermaid.live/edit#base64:" + base64_str)
