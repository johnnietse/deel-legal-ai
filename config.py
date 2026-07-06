# Deel Lab Legal AI System - Configuration
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Keys - Environment variables only, no hardcoded values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# Validate API keys at runtime
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")
    
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is required")

# Gemini API Configuration
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_CHAT_MODEL = "gemini-2.0-flash"

# Pinecone Configuration
PINECONE_INDEX_NAME = "deel-legal-cases"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_DIMENSION = 768  # Gemini embedding dimension
PINECONE_METRIC = "cosine"

# RAG Pipeline Configuration
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 50  # tokens (10% overlap for legal context preservation)
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 16  # seconds between CanLII requests

# Multi-Hop RAG Configuration
MULTI_HOP_MAX_HOPS = 5
MULTI_HOP_COMPLETENESS_THRESHOLD = 0.8  # Stop when evidence is >= 80% complete
MULTI_HOP_MIN_NEW_INFO_TOKENS = 50  # Minimum new info to justify another hop

# Knowledge Graph Configuration
KNOWLEDGE_GRAPH_PATH = DATA_DIR / "legal_knowledge_graph.json"
KG_MAX_SUBGRAPH_DEPTH = 3
KG_ENTITY_TYPES = ["Case", "Court", "Judge", "LegalTest", "Factor", "Jurisdiction", "Party", "Statute"]
KG_RELATION_TYPES = [
    "cites", "applies_test", "involves_factor", "decided_by",
    "supports_classification", "overrules", "distinguishes",
    "enacted_by", "amends", "interprets"
]

# MCTS Legal Reasoning Agent Configuration
MCTS_N_SIMULATIONS = 50
MCTS_EXPLORATION_CONSTANT = 1.414  # UCB1 sqrt(2)
MCTS_MAX_DEPTH = 6
MCTS_MIN_SCORE_THRESHOLD = 0.2  # Prune branches below this score
MCTS_REWARD_WEIGHTS = {
    "precedent_alignment": 0.35,
    "factor_completeness": 0.25,
    "logical_consistency": 0.25,
    "evidence_strength": 0.15,
}

# Evaluation Framework Configuration
EVAL_DIR = BASE_DIR / "evaluation"
EVAL_RESULTS_DIR = BASE_DIR / "evaluation_results"
EVAL_RESULTS_DIR.mkdir(exist_ok=True)
EVAL_DEFAULT_N_CASES = 50
EVAL_DIMENSIONS = [
    "factor_identification",
    "legal_reasoning_quality",
    "citation_accuracy",
    "risk_assessment",
    "completeness",
    "hedging_appropriateness",
]

# LLM Judge Configuration
JUDGE_MODEL = "gemini-2.0-flash"  # Model used for judging
JUDGE_TEMPERATURE = 0.1  # Low temperature for consistent scoring
JUDGE_POSITION_SWAP_TRIALS = 2  # Number of swap trials for position debiasing

# CanLII Scraper Configuration
CANLII_BASE_URL = "https://www.canlii.org"
CANLII_PDF_DOWNLOAD_DIR = DATA_DIR / "canlii_pdfs"
CANLII_PDF_DOWNLOAD_DIR.mkdir(exist_ok=True)

# ML Classifier Configuration
ML_MODEL_PATH = MODELS_DIR / "worker_classifier.joblib"
ML_FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
ML_TEST_SIZE = 0.2
ML_RANDOM_STATE = 42

# Employment Cases Dataset
EMPLOYMENT_CASES_CSV = DATA_DIR / "employment_cases_large.csv"

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# New module dependencies check
NETWORKX_AVAILABLE = True
try:
    import networkx
except ImportError:
    NETWORKX_AVAILABLE = False

# =============================================================================
# ByteDance RAG Enhancements Configuration
# =============================================================================

# --- Elasticsearch BM25 (ByteDance §5.2) ---
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_CLOUD_ID = os.getenv("ELASTICSEARCH_CLOUD_ID", "")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
ELASTICSEARCH_INDEX_NAME = "deel-legal-chunks"

# --- Milvus Vector Store (ByteDance ByteVectorDB alternative) ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_COLLECTION_NAME = "deel_legal_vectors"

# --- Vector Store Backend Selection ---
# Options: "pinecone" (managed), "milvus" (self-hosted)
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "pinecone")

# --- HNSW Index Tuning (ByteDance §4.2.1) ---
# Options: "development", "production", "high_recall", "billion_scale"
HNSW_PRESET = os.getenv("HNSW_PRESET", "production")

# --- Hybrid Search Configuration (ByteDance §5.2) ---
HYBRID_SEARCH_ENABLED = True
HYBRID_FUSION_METHOD = "rrf"        # "rrf" (Reciprocal Rank Fusion) or "weighted"
HYBRID_MMR_LAMBDA = 0.7             # Diversity vs relevance trade-off (§5.3.3)
HYBRID_DEFAULT_TOP_K = 5
# Query-type adaptive weights (ByteDance §5.2.3)
HYBRID_KEYWORD_WEIGHTS = (0.7, 0.3)  # (bm25, vector) for citation queries
HYBRID_SEMANTIC_WEIGHTS = (0.2, 0.8)  # (bm25, vector) for conceptual queries
HYBRID_BALANCED_WEIGHTS = (0.4, 0.6)  # (bm25, vector) for mixed queries

# --- BM25 Backend Selection ---
# Options: "elasticsearch" (production), "local" (development)
BM25_BACKEND = os.getenv("BM25_BACKEND", "elasticsearch")

# --- Semantic Chunking Configuration (ByteDance §4.1.2) ---
SEMANTIC_CHUNKING_ENABLED = True
SEMANTIC_CHUNK_MAX_TOKENS = 512
SEMANTIC_CHUNK_MIN_TOKENS = 50
SEMANTIC_CHUNK_NARRATIVE_TARGET = 384    # Narrative reasoning text
SEMANTIC_CHUNK_STRUCTURED_TARGET = 128   # Statutes, lists, tables

# --- Multi-Granularity Vectors (ByteDance §4.1.2) ---
MULTI_GRANULARITY_ENABLED = False  # Enable when ready to re-index
DOCUMENT_SUMMARY_NAMESPACE = "legal_cases_docs"
CHUNK_NAMESPACE = "legal_cases"

# --- Prompt Template Configuration (ByteDance §6.2) ---
PROMPT_MAX_SOURCES = 5               # Max sources to inject into prompt
PROMPT_MIN_SIMILARITY = 0.0          # Min score to keep a source in prompt

# --- Confidence Gate (ByteDance §6.3.1) ---
CONFIDENCE_GATE_ENABLED = True
CONFIDENCE_REFUSE_THRESHOLD = 0.3    # Below → refuse to answer
CONFIDENCE_HEDGE_THRESHOLD = 0.5     # Below → add hedging language

# --- Query & Embedding Cache (ByteDance §6.4.2) ---
CACHE_ENABLED = True
CACHE_DIR = str(DATA_DIR / "cache")
CACHE_EMBEDDING_MAXSIZE = 2000
CACHE_EMBEDDING_TTL = 3600           # 1 hour
CACHE_RETRIEVAL_MAXSIZE = 500
CACHE_RETRIEVAL_TTL = 600            # 10 minutes
CACHE_RESPONSE_MAXSIZE = 500
CACHE_RESPONSE_TTL = 300             # 5 minutes

# --- Feedback System (ByteDance §6.3.3) ---
FEEDBACK_STORE_PATH = str(DATA_DIR / "feedback.jsonl")
FEEDBACK_MIN_WRONG_FOR_FLAG = 3      # Flag queries with N+ "wrong" ratings

# --- Pipeline Metrics (ByteDance §8.1) ---
METRICS_LOG_DIR = str(LOGS_DIR / "metrics")
METRICS_MAX_IN_MEMORY = 10000

# --- Model Optimisation (ByteDance §7.2–7.4) ---
# LoRA fine-tuning
LORA_BASE_MODEL = os.getenv("BASE_LLM_MODEL", "google/gemma-2-2b-it")
LORA_OUTPUT_DIR = str(MODELS_DIR / "lora_legal")
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TRAIN_DATA = str(DATA_DIR / "training" / "lora_train.jsonl")
LORA_EVAL_DATA = str(DATA_DIR / "training" / "lora_eval.jsonl")

# Quantisation
QUANTISATION_BITS = 8                # 4 or 8
QUANTISATION_METHOD = "bitsandbytes"  # "bitsandbytes", "gptq", "awq"

# Knowledge distillation
DISTILLATION_TEACHER = "gemini-2.0-flash"
DISTILLATION_STUDENT = "google/gemma-2-2b-it"
DISTILLATION_SAMPLES = 5000

# --- GPU Auto-Scaling (ByteDance §8.2) ---
AUTOSCALE_MIN_REPLICAS = 1
AUTOSCALE_MAX_REPLICAS = 8
AUTOSCALE_TARGET_QPS = 10
AUTOSCALE_SCALE_UP_THRESHOLD = 0.8
AUTOSCALE_SCALE_DOWN_THRESHOLD = 0.3
AUTOSCALE_COOLDOWN_SECONDS = 300

# --- Cross-Region Deployment (ByteDance §8.3) ---
PRIMARY_REGION = os.getenv("PRIMARY_REGION", "us-east-1")
REPLICA_REGIONS = os.getenv("REPLICA_REGIONS", "eu-west-1,ap-southeast-1").split(",")
REPLICATION_STRATEGY = "async"       # "async" or "sync"

