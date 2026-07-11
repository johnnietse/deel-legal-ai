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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CHUNK_NAMESPACE
from ml_classifier import WorkerClassificationAPI, ClassificationRequest
from rag_pipeline import LegalRAGQuery
from config import LOG_FORMAT, LOG_LEVEL
from db.database import init_db, close_db
from api.auth import get_optional_user, get_api_key_user

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Lifespan handler (FastAPI 2.6+) ──────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize DB on startup, dispose on shutdown."""
    logger.info("Starting up — initializing database…")
    try:
        await init_db()
        logger.info("Database tables ready.")
    except Exception as e:
        logger.warning("Database init failed (non-fatal): %s", e)
    yield
    logger.info("Shutting down — closing database…")
    await close_db()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Deel Lab Legal AI API",
    description="Legal Research Assistant API for worker classification and case law retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
import os as _os
_ALLOWED_ORIGINS = _os.getenv("CORS_ORIGINS", "").split(",") if _os.getenv("CORS_ORIGINS") else []
_DEFAULT_ORIGINS = [
    "https://deel.ai",
    "https://app.deel.ai",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://[::1]:5173",
    "http://[::1]:5174",
    "https://green-field-0674bf00f.7.azurestaticapps.net",
    "https://*.7.azurestaticapps.net",
    "http://openjustice-api.eastus2.azurecontainer.io:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + _ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Add rate limiting middleware for SaaS endpoints
from api.middleware import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

# Initialize services (lazy loading)
rag_query = None
classification_api = None


def get_rag_query():
    """Lazy load RAG query interface.
    
    Uses VECTOR_STORE_BACKEND from .env (can be 'pinecone', 'milvus',
    or 'both' for DualVectorStore with Pinecone primary + Milvus failover).
    """
    global rag_query
    if rag_query is None:
        try:
            from config import PINECONE_INDEX_NAME, VECTOR_STORE_BACKEND
            from rag_pipeline.vector_store import create_vector_store
            rag_query = LegalRAGQuery(
                vector_store=create_vector_store(
                    backend=VECTOR_STORE_BACKEND,
                    index_name=PINECONE_INDEX_NAME,
                )
            )
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
    verify: bool = Field(default=False, description="Run post-hoc verification to check for hallucinations")


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: List[Dict[str, Any]]
    verification: Optional[Dict[str, Any]] = None


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
async def rag_query_endpoint(
    request: Request,
    query_data: RAGQueryRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Query the legal knowledge base using RAG.
    
    Returns relevant legal information with sources and citations.
    """
    effective_user = user or api_user
    
    # Input validation
    if not query_data.question or len(query_data.question.strip()) < 10:
        raise HTTPException(status_code=400, detail="Question must be at least 10 characters long")
    
    # Unauthenticated: stricter limits
    if not effective_user:
        if len(query_data.question) > 1000:
            raise HTTPException(status_code=400, detail="Query too long for unauthenticated access (max 1000 chars)")
    else:
        if len(query_data.question) > 5000:
            raise HTTPException(status_code=400, detail="Question too long (max 5000 characters)")
    
    rag = get_rag_query()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    
    try:
        filter_dict = {"jurisdiction": query_data.jurisdiction} if query_data.jurisdiction else None
        
        # Pass user_id to RAG query for metrics tracking
        user_id = effective_user.get("sub") if effective_user else None
        
        response = rag.query(
            question=query_data.question,
            top_k=query_data.top_k,
            filter=filter_dict,
            verify=query_data.verify,
            namespace=CHUNK_NAMESPACE,
            user_id=user_id,
        )
        
        return RAGQueryResponse(
            query=response.query,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            verification=response.verification
        )
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify", response_model=ClassificationResponseModel)
async def classify_worker(
    request: Request,
    classify_data: ClassificationRequestModel,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Classify a worker as employee or independent contractor.
    
    Uses Random Forest classifier trained on 700+ employment law cases.
    """
    effective_user = user or api_user
    
    clf = get_classification_api()
    if not clf or not clf.model.is_trained:
        raise HTTPException(
            status_code=503, 
            detail="Classification service unavailable. Model needs training."
        )
    
    try:
        # Convert to internal request format
        internal_request = ClassificationRequest(
            supervision_review=classify_data.supervision_review,
            ability_hire=classify_data.ability_hire,
            delegation_tasks=classify_data.delegation_tasks,
            ownership_tools=classify_data.ownership_tools,
            chance_profit=classify_data.chance_profit,
            risk_loss=classify_data.risk_loss,
            exclusivity_services=classify_data.exclusivity_services,
            work_hours_setter=classify_data.work_hours_setter,
            work_location=classify_data.work_location,
            uniform_required=classify_data.uniform_required
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
async def classify_workers_batch(
    request: Request,
    requests: List[ClassificationRequestModel],
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Classify multiple workers in batch.
    """
    effective_user = user or api_user
    
    clf = get_classification_api()
    if not clf or not clf.model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Classification service unavailable. Model needs training."
        )
    
    results = []
    for classify_data in requests:
        try:
            result = await classify_worker(request, classify_data, user, api_user)
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
# Advanced RAG Endpoints (Modules 1-2)
# =====================

class MultiHopQueryRequest(BaseModel):
    question: str = Field(..., description="Complex legal question for multi-hop retrieval")
    max_hops: int = Field(default=5, ge=1, le=10, description="Maximum retrieval hops")
    top_k_per_hop: int = Field(default=3, ge=1, le=10, description="Documents per hop")
    jurisdiction: Optional[str] = Field(default=None, description="Jurisdiction filter")
    verify: bool = Field(default=False, description="Run post-hoc verification to check for hallucinations")


class MultiHopQueryResponse(BaseModel):
    query: str
    answer: str
    confidence: str
    sources: List[Dict[str, Any]]
    total_hops: int = 0
    retrieval_mode: str = "multi_hop"
    verification: Optional[Dict[str, Any]] = None


@app.post("/rag/query/multi-hop", response_model=MultiHopQueryResponse)
async def rag_multi_hop_query(
    request: Request,
    query_data: MultiHopQueryRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Query using multi-hop retrieval for complex legal questions.
    
    Performs iterative retrieve-read-reason cycles to build complete
    evidence chains across multiple document nodes.
    """
    effective_user = user or api_user
    
    # Unauthenticated: stricter limits
    if not effective_user:
        if len(query_data.question) > 1000:
            raise HTTPException(status_code=400, detail="Query too long for unauthenticated access (max 1000 chars)")
    else:
        if len(query_data.question) > 5000:
            raise HTTPException(status_code=400, detail="Question too long (max 5000 characters)")
    
    rag = get_rag_query()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    
    try:
        filter_dict = {"jurisdiction": query_data.jurisdiction} if query_data.jurisdiction else None
        
        user_id = effective_user.get("sub") if effective_user else None
        
        response = rag.query_multi_hop(
            question=query_data.question,
            max_hops=query_data.max_hops,
            top_k_per_hop=query_data.top_k_per_hop,
            filter=filter_dict,
            verify=query_data.verify,
            user_id=user_id,
        )
        
        return MultiHopQueryResponse(
            query=response.query,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            total_hops=len(response.sources),
            retrieval_mode="multi_hop",
            verification=response.verification,
        )
        
    except Exception as e:
        logger.error(f"Multi-hop query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query/smart", response_model=MultiHopQueryResponse)
async def rag_smart_query(
    request: Request,
    query_data: RAGQueryRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Smart query that auto-routes between single-hop and multi-hop
    based on question complexity analysis.
    """
    effective_user = user or api_user
    
    # Unauthenticated: stricter limits
    if not effective_user:
        if len(query_data.question) > 1000:
            raise HTTPException(status_code=400, detail="Query too long for unauthenticated access (max 1000 chars)")
    else:
        if len(query_data.question) > 5000:
            raise HTTPException(status_code=400, detail="Question too long (max 5000 characters)")
    
    rag = get_rag_query()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    
    try:
        filter_dict = {"jurisdiction": query_data.jurisdiction} if query_data.jurisdiction else None
        
        user_id = effective_user.get("sub") if effective_user else None
        
        response = rag.query_smart(
            question=query_data.question,
            filter=filter_dict,
            verify=query_data.verify,
            user_id=user_id,
        )
        
        return MultiHopQueryResponse(
            query=response.query,
            answer=response.answer,
            confidence=response.confidence,
            sources=response.sources,
            retrieval_mode="smart_auto",
            verification=response.verification,
        )
        
    except Exception as e:
        logger.error(f"Smart query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class VerifyRequest(BaseModel):
    answer: str = Field(..., description="Draft answer to verify")
    sources: List[Dict[str, Any]] = Field(..., description="Sources retrieved from Pinecone/Graph")

@app.post("/rag/verify")
async def verify_standalone(
    request: Request,
    verify_data: VerifyRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Standalone endpoint to fact-check an arbitrary answer against sources
    using the NLI Verifier and assumption extraction.
    """
    effective_user = user or api_user
    
    rag = get_rag_query()
    if not rag or not hasattr(rag, 'verifier'):
        raise HTTPException(status_code=503, detail="Verification service unavailable")
        
    try:
        verification = rag.verifier.verify_grounding(verify_data.answer, verify_data.sources)
        return verification.to_dict()
    except Exception as e:
        logger.error(f"Standalone verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# ByteDance Enhancements — Feedback & Observability
# =====================

class FeedbackRequest(BaseModel):
    query_id: str = Field(default="", description="ID of the query being rated")
    query_text: str = Field(..., description="The original question")
    answer_text: str = Field(default="", description="The answer that was given")
    rating: str = Field(..., description="Rating: 'useful', 'not_useful', or 'wrong'")
    error_type: Optional[str] = Field(
        default=None,
        description="Error category: 'data_error', 'incomplete', 'off_topic', 'hallucination'"
    )
    comment: Optional[str] = Field(default=None, description="Free-text comment")


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a RAG response.
    
    ByteDance §6.3.3: Collect "useful/not useful/wrong" feedback per response.
    Wrong-feedback triggers root cause analysis. Good answers go into the
    "excellent answer library" for few-shot examples and fine-tuning.
    """
    from rag_pipeline.feedback_analyzer import FeedbackStore, FeedbackEntry
    
    valid_ratings = {"useful", "not_useful", "wrong"}
    if request.rating not in valid_ratings:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rating. Must be one of: {valid_ratings}"
        )
    
    valid_errors = {"data_error", "incomplete", "off_topic", "hallucination", None}
    if request.error_type not in valid_errors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid error_type. Must be one of: {valid_errors}"
        )
    
    try:
        store = FeedbackStore()
        entry = FeedbackEntry(
            query_id=request.query_id,
            query_text=request.query_text,
            answer_text=request.answer_text,
            rating=request.rating,
            error_type=request.error_type,
            comment=request.comment,
        )
        store.record(entry)
        
        return {
            "status": "recorded",
            "query_id": request.query_id,
            "rating": request.rating,
        }
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/summary")
async def get_feedback_summary():
    """
    Get a summary of collected user feedback.
    
    Includes: total count, rating distribution, error type breakdown,
    flagged queries, and root cause analysis.
    """
    from rag_pipeline.feedback_analyzer import FeedbackStore, FeedbackAnalyzer
    
    try:
        analyzer = FeedbackAnalyzer()
        summary = analyzer.summary()
        flagged = analyzer.get_flagged_queries()
        root_cause = analyzer.root_cause_breakdown()
        
        return {
            "summary": summary,
            "flagged_queries": flagged[:10],
            "root_cause": root_cause,
        }
    except Exception as e:
        logger.error(f"Feedback summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/stats")
async def get_pipeline_stats():
    """
    Get full pipeline statistics (ByteDance §8.1).
    
    Returns: latency percentiles, quality scores, cost estimates,
    cache hit rates, retrieval mode distribution.
    """
    rag = get_rag_query()
    if not rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable")
    
    try:
        return rag.get_pipeline_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# MCTS Reasoning Endpoint (Module 3)
# =====================

class MCTSClassificationRequest(BaseModel):
    facts: str = Field(..., description="Description of the working relationship")
    n_simulations: int = Field(default=20, ge=5, le=100, description="MCTS simulations")


class MCTSClassificationResponse(BaseModel):
    classification: str
    confidence: float
    factor_analysis: Dict[str, Dict[str, Any]]
    reasoning_text: str
    tree_statistics: Dict[str, Any]
    duration_ms: float


@app.post("/classify/reasoning", response_model=MCTSClassificationResponse)
async def classify_with_reasoning(
    request: Request,
    classify_data: MCTSClassificationRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Classify worker using MCTS-based legal reasoning.
    
    Explores a tree of classification hypotheses using Monte Carlo
    Tree Search, scoring each against RAG-retrieved precedents.
    Returns full reasoning trace with per-factor analysis.
    """
    effective_user = user or api_user
    
    from rag_pipeline.legal_reasoning_agent import LegalReasoningAgent
    
    try:
        agent = LegalReasoningAgent(n_simulations=classify_data.n_simulations)
        result = agent.classify_with_reasoning(classify_data.facts)
        
        return MCTSClassificationResponse(
            classification=result.classification,
            confidence=result.confidence,
            factor_analysis=result.factor_analysis,
            reasoning_text=result.full_reasoning_text,
            tree_statistics=result.tree_statistics,
            duration_ms=result.duration_ms,
        )
        
    except Exception as e:
        logger.error(f"MCTS classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Evaluation Endpoints (Modules 4-5)
# =====================

class BenchmarkRequest(BaseModel):
    n_cases: int = Field(default=10, ge=1, le=100, description="Number of test cases")
    seed: int = Field(default=42, description="Random seed for reproducibility")


@app.post("/evaluate/generate-suite")
async def generate_benchmark_suite(
    request: Request,
    benchmark_data: BenchmarkRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Generate a dynamic, anti-contamination legal evaluation test suite.
    
    Each test case is parameterized with randomized names, companies,
    and amounts while preserving legal logic. Same seed = same suite.
    """
    effective_user = user or api_user
    
    from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
    
    try:
        generator = LegalBenchmarkGenerator(base_seed=benchmark_data.seed)
        suite = generator.generate_suite(n_cases=benchmark_data.n_cases)
        suite.save()
        
        return {
            "suite_id": suite.suite_id,
            "n_cases": suite.n_cases,
            "difficulty_distribution": suite.difficulty_distribution,
            "cases_preview": [
                {
                    "case_id": c.case_id,
                    "difficulty": c.difficulty,
                    "expected": c.expected_classification,
                    "scenario_preview": c.scenario[:200] + "...",
                }
                for c in suite.cases[:3]
            ],
        }
        
    except Exception as e:
        logger.error(f"Benchmark generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class JudgeRequest(BaseModel):
    question: str = Field(..., description="The question asked")
    response: str = Field(..., description="The AI response to evaluate")
    reference: Optional[str] = Field(default=None, description="Optional reference answer")


@app.post("/evaluate/judge")
async def judge_response(
    request: Request,
    judge_data: JudgeRequest,
    user: Optional[dict] = Depends(get_optional_user),
    api_user: Optional[dict] = Depends(get_api_key_user),
):
    """
    Score a legal AI response using the debiased LLM judge.
    
    Returns component-level scores with bias mitigation applied
    (rubric decomposition, length normalization).
    """
    effective_user = user or api_user
    
    from evaluation.llm_judge import DebiasedLegalJudge
    
    try:
        judge = DebiasedLegalJudge()
        result = judge.score(
            question=judge_data.question,
            response=judge_data.response,
            reference=judge_data.reference,
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Judge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# OpenJustice.ai SaaS Endpoints (via aggregated router)
# =====================

from api.router import api_router
app.include_router(api_router)

# =====================
# Frontend Static Files (OpenJustice.ai SaaS)
# =====================

# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse

# FRONTEND_DIST = Path(__file__).parent.parent / "openjustice-frontend" / "dist"

# if FRONTEND_DIST.exists():
#     # Serve static assets (JS, CSS, images)
#     app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
#     
#     # Serve favicon and other root files
#     @app.get("/favicon.svg")
#     async def favicon():
#         return FileResponse(FRONTEND_DIST / "favicon.svg")
#     
#     @app.get("/icons.svg")
#     async def icons():
#         return FileResponse(FRONTEND_DIST / "icons.svg")
#     
#     # Catch-all route for SPA - serves index.html for all non-API routes
#     @app.get("/{full_path:path}")
#     async def spa_catchall(full_path: str):
#         # Don't intercept API routes
#         if full_path.startswith("api/"):
#             raise HTTPException(status_code=404, detail="Not found")
#         return FileResponse(FRONTEND_DIST / "index.html")
# else:
#     logger.warning(f"Frontend build not found at {FRONTEND_DIST}. Run 'npm run build' in openjustice-frontend/")

# =====================
# Run Server
# =====================

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
