# Deel Lab Legal AI System - FastAPI Service
"""
REST API for the Legal AI System.

Endpoints:
- /health - Health check
- /rag/query - RAG query for legal Q&A
- /classify - Worker classification prediction
- /classify/batch - Batch worker classification
- /model/info - Model information
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline import LegalRAGPipeline, LegalRAGQuery
from ml_classifier import WorkerClassificationAPI, ClassificationRequest
from config import LOG_FORMAT, LOG_LEVEL

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deel Lab Legal AI API",
    description="Legal Research Assistant API for worker classification and case law retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy loading)
rag_query = None
classification_api = None


def get_rag_query():
    """Lazy load RAG query interface"""
    global rag_query
    if rag_query is None:
        try:
            rag_query = LegalRAGQuery()
        except Exception as e:
            logger.error(f"Failed to initialize RAG query: {e}")
    return rag_query


def get_classification_api():
    """Lazy load classification API"""
    global classification_api
    if classification_api is None:
        try:
            classification_api = WorkerClassificationAPI()
        except Exception as e:
            logger.error(f"Failed to initialize classification API: {e}")
    return classification_api


# =====================
# Request/Response Models
# =====================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]


class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="Legal question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    jurisdiction: Optional[str] = Field(default=None, description="Filter by jurisdiction (e.g., 'ON', 'BC')")


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: List[Dict[str, Any]]


class ClassificationRequestModel(BaseModel):
    supervision_review: str = Field(default="Unknown", description="Degree of supervision")
    ability_hire: str = Field(default="Unknown", description="Can hire employees?")
    delegation_tasks: str = Field(default="Unknown", description="Can delegate tasks?")
    ownership_tools: str = Field(default="Unknown", description="Who owns tools?")
    chance_profit: str = Field(default="Unknown", description="Chance of profit?")
    risk_loss: str = Field(default="Unknown", description="Risk of loss?")
    exclusivity_services: str = Field(default="Unknown", description="Exclusive services?")
    work_hours_setter: str = Field(default="Unknown", description="Who sets hours?")
    work_location: str = Field(default="Unknown", description="Work location?")
    uniform_required: str = Field(default="Unknown", description="Uniform required?")


class ClassificationResponseModel(BaseModel):
    prediction: str
    confidence: float
    is_employee: bool
    class_probabilities: Dict[str, float]
    contributing_factors: List[Dict[str, Any]]
    legal_interpretation: str


class ModelInfoResponse(BaseModel):
    rag_status: str
    classifier_status: str
    classifier_accuracy: Optional[float]
    n_training_samples: Optional[int]
    top_features: Optional[List[Dict[str, Any]]]


# =====================
# API Endpoints
# =====================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs"""
    return {"message": "Deel Lab Legal AI API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health of all services"""
    services = {}
    
    # Check RAG
    rag = get_rag_query()
    services["rag"] = "available" if rag else "unavailable"
    
    # Check classifier
    clf = get_classification_api()
    if clf and clf.model.is_trained:
        services["classifier"] = "available"
    else:
        services["classifier"] = "unavailable"
    
    return HealthResponse(
        status="healthy" if all(s == "available" for s in services.values()) else "degraded",
        timestamp=datetime.now().isoformat(),
        services=services
    )


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query_endpoint(request: RAGQueryRequest):
    """
    Query the legal knowledge base using RAG.
    
    Returns relevant legal information with sources and citations.
    """
    rag = get_rag_query()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    
    try:
        filter_dict = {"jurisdiction": request.jurisdiction} if request.jurisdiction else None
        
        response = rag.query(
            question=request.question,
            top_k=request.top_k,
            filter=filter_dict
        )
        
        return RAGQueryResponse(
            query=response.query,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources
        )
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassificationResponseModel)
async def classify_worker(request: ClassificationRequestModel):
    """
    Classify a worker as employee or independent contractor.
    
    Uses Random Forest classifier trained on 700+ employment law cases.
    """
    clf = get_classification_api()
    if not clf or not clf.model.is_trained:
        raise HTTPException(
            status_code=503, 
            detail="Classification service unavailable. Model needs training."
        )
    
    try:
        # Convert to internal request format
        internal_request = ClassificationRequest(
            supervision_review=request.supervision_review,
            ability_hire=request.ability_hire,
            delegation_tasks=request.delegation_tasks,
            ownership_tools=request.ownership_tools,
            chance_profit=request.chance_profit,
            risk_loss=request.risk_loss,
            exclusivity_services=request.exclusivity_services,
            work_hours_setter=request.work_hours_setter,
            work_location=request.work_location,
            uniform_required=request.uniform_required
        )
        
        response = clf.classify(internal_request)
        
        return ClassificationResponseModel(
            prediction=response.prediction,
            confidence=response.confidence,
            is_employee=response.is_employee,
            class_probabilities=response.class_probabilities,
            contributing_factors=response.contributing_factors,
            legal_interpretation=response.legal_interpretation
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=List[ClassificationResponseModel])
async def classify_workers_batch(requests: List[ClassificationRequestModel]):
    """
    Classify multiple workers in batch.
    """
    clf = get_classification_api()
    if not clf or not clf.model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Classification service unavailable. Model needs training."
        )
    
    results = []
    for request in requests:
        try:
            result = await classify_worker(request)
            results.append(result)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return results


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about available models.
    """
    rag = get_rag_query()
    clf = get_classification_api()
    
    response = ModelInfoResponse(
        rag_status="available" if rag else "unavailable",
        classifier_status="available" if clf and clf.model.is_trained else "unavailable",
        classifier_accuracy=None,
        n_training_samples=None,
        top_features=None
    )
    
    if clf and clf.model.is_trained:
        info = clf.get_model_info()
        response.classifier_accuracy = info.get("accuracy")
        response.n_training_samples = info.get("n_training_samples")
        response.top_features = info.get("top_features")
    
    return response


@app.get("/classify/factors")
async def get_classification_factors():
    """
    Get definitions of worker classification factors.
    """
    clf = get_classification_api()
    if clf:
        return clf.get_feature_definitions()
    return {}


# =====================
# Run Server
# =====================

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
