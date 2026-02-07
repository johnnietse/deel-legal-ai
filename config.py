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

# API Keys
# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set in environment variables or .env file")
    
if not PINECONE_API_KEY:
    print("WARNING: PINECONE_API_KEY not set in environment variables or .env file")

# Gemini API Configuration
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_EMBEDDING_MODEL = "text-embedding-004"
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
