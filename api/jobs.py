# OpenJustice.ai - Background Job Manager
"""
Long-running RAG/MCTS queries run as background jobs so the event loop
stays responsive while free-tier providers throttle individual calls.

Endpoints:
- POST /rag/query/job           - submit a query, returns job_id immediately
- GET  /rag/query/job/{id}      - poll status + result

Jobs are persisted to disk (data/rag_jobs.json) so queued/running jobs
survive a process restart. Completed jobs are retained for
JOB_RETENTION_SECONDS, then pruned.

Auth: optional. Submit/poll accept optional user (JWT) or API key. Jobs
created by an authenticated user are only readable by that same user;
anonymous jobs are guarded by an unguessable job_id.
"""

import sys
import uuid
import time
import json
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from config import DATA_DIR
from api.auth import get_optional_user, get_api_key_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Jobs"])

JOB_RETENTION_SECONDS = 3600  # keep completed jobs 1h for polling/late fetch
JOB_STORE_PATH = DATA_DIR / "rag_jobs.json"
MAX_ACTIVE_JOBS = 20          # global cap on queued+running
MAX_ACTIVE_PER_USER = 3       # per-user cap on queued+running
JOB_SUBMIT_COOLDOWN = 3       # min seconds between job submits per user

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_active_count = 0
_last_submit_ts: Dict[str, float] = {}


class JobSubmitRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=5000, description="Legal question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    n_simulations: Optional[int] = Field(default=None, ge=2, le=50, description="MCTS simulations (optional override)")
    jurisdiction: Optional[str] = Field(default=None, description="Filter by jurisdiction")
    court: Optional[str] = Field(default=None, description="Filter by court (e.g. 'ONSC')")
    statute: Optional[str] = Field(default=None, description="Filter by statute (e.g. 'ESA')")
    legal_section: Optional[str] = Field(default=None, description="Filter by legal section")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata filters (supports $in/$eq)")
    mode: Optional[str] = Field(default=None, description="Retrieval mode: 'graphrag' or None for auto")


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    question: str
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prune_locked():
    """Drop expired jobs (called with _lock held). Unfinished jobs kept."""
    cutoff = time.time() - JOB_RETENTION_SECONDS
    for jid, jb in list(_jobs.items()):
        finished_ts = jb.get("finished_ts")
        if finished_ts is not None and finished_ts < cutoff:
            _jobs.pop(jid, None)


def _persist():
    """Write the job store to disk. Idempotent, best-effort."""
    try:
        JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        JOB_STORE_PATH.write_text(
            json.dumps(list(_jobs.values()), indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Job store persist failed: %s", e)


def _load_jobs():
    """Restore jobs persisted by a previous process."""
    global _active_count
    if not JOB_STORE_PATH.exists():
        return
    try:
        stored = json.loads(JOB_STORE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Job store load failed: %s", e)
        return
    for jb in stored:
        if not isinstance(jb, dict) or not jb.get("job_id"):
            continue
        # Only reload unfinished jobs; finished ones aged out anyway.
        if jb.get("status") in ("queued", "running"):
            jb["status"] = "queued"  # restart: re-run from scratch
            jb["started_at"] = None
            jb["started_ts"] = None
            jb["finished_at"] = None
            jb["finished_ts"] = None
            jb["result"] = None
            jb["error"] = None
            jb["restarted"] = True
            _jobs[jb["job_id"]] = jb
            _active_count += 1
    if _jobs:
        logger.info("Restored %d unfinished job(s) from disk", len(_jobs))


_load_jobs()


class _RateLimitExceeded(Exception):
    pass


def _active_for_user_locked(user_id: str) -> int:
    return sum(
        1 for jb in _jobs.values()
        if jb.get("user_id") == user_id
        and jb.get("status") in ("queued", "running")
    )


def _check_submit_allowed(user_key: str):
    """Enforce global + per-user caps and a submit cooldown.

    Raises _RateLimitExceeded with the retry_after seconds on failure.
    """
    with _lock:
        _prune_locked()
        if _active_count >= MAX_ACTIVE_JOBS:
            raise _RateLimitExceeded(30)
        if user_key is not None:
            if _active_for_user_locked(user_key) >= MAX_ACTIVE_PER_USER:
                raise _RateLimitExceeded(max(10, JOB_SUBMIT_COOLDOWN * 2))
        # Cooldown applies to authenticated users and (globally) anon.
        key = user_key or "__anon__"
        last = _last_submit_ts.get(key, 0.0)
        wait = JOB_SUBMIT_COOLDOWN - (time.time() - last)
        if wait > 0:
            raise _RateLimitExceeded(round(wait, 1))
        _last_submit_ts[key] = time.time()


def _result_to_dict(response) -> Dict[str, Any]:
    return {
        "query": response.query,
        "answer": response.answer,
        "confidence": response.confidence,
        "sources": response.sources,
        "verification": getattr(response, "verification", None),
        "retrieval_mode": getattr(response, "retrieval_mode", "single_hop"),
        "metrics": getattr(response, "metrics", None),
        "status": getattr(response, "status", "ok"),
        "error_type": getattr(response, "error_type", None),
        "error_message": getattr(response, "error_message", None),
    }


def _run_job(job_id: str, payload: Dict[str, Any]):
    """Execute a query in a background thread. Never touches the event loop."""
    global _active_count
    job = _jobs.get(job_id)
    if not job:
        return

    with _lock:
        job["status"] = "running"
        job["started_at"] = _now_iso()
        job["started_ts"] = time.time()
        _persist()

    try:
        from config import CHUNK_NAMESPACE
        from rag_pipeline.services import build_rag_query
        rag = build_rag_query()

        kwargs = {
            "question": payload["question"],
            "verify": payload.get("verify", False),
            "namespace": CHUNK_NAMESPACE,
        }
        # Build merged metadata filter from explicit fields + generic filters dict
        filter_dict: Dict[str, Any] = {}
        for field in ("jurisdiction", "court", "statute", "legal_section"):
            val = payload.get(field)
            if val:
                filter_dict[field] = val
        extra = payload.get("filters") or {}
        if isinstance(extra, dict):
            filter_dict.update(extra)
        if filter_dict:
            kwargs["filter"] = filter_dict

        mode = payload.get("mode")
        if mode == "graphrag":
            kwargs["mode"] = "graphrag"

        # query_smart routes automatically (single-hop / MCTS / multi-hop).
        # n_simulations caps the MCTS budget when the client needs faster
        # free-tier turnaround; query_smart keeps adaptive routing otherwise.
        if payload.get("n_simulations"):
            response = rag.query_reasoned(
                **kwargs, n_simulations=payload["n_simulations"]
            )
        else:
            response = rag.query_smart(**kwargs)

        with _lock:
            job["status"] = "done"
            job["result"] = _result_to_dict(response)
            job["finished_at"] = _now_iso()
            job["finished_ts"] = time.time()
            _active_count = max(0, _active_count - 1)
            _persist()
    except Exception as e:
        logger.error("Background job %s failed: %s", job_id, e)
        with _lock:
            job["status"] = "error"
            job["error"] = str(e)
            job["finished_at"] = _now_iso()
            job["finished_ts"] = time.time()
            _active_count = max(0, _active_count - 1)
            _persist()


@router.post("/rag/query/job", response_model=JobSubmitResponse)
async def submit_job(
    request: Request,
    payload: JobSubmitRequest,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
    api_user: Optional[Dict[str, Any]] = Depends(get_api_key_user),
):
    """Submit a RAG query as a background job. Returns immediately."""
    global _active_count
    effective_user = user or api_user
    user_id = effective_user.get("user_id") or effective_user.get("sub") if effective_user else None

    # Rate limiting: global cap, per-user cap, and submit cooldown.
    try:
        _check_submit_allowed(user_id)
    except _RateLimitExceeded as e:
        retry = e.args[0] if e.args else JOB_SUBMIT_COOLDOWN
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Too many jobs in flight. Try again shortly.",
                "retry_after_seconds": retry,
            },
            headers={"Retry-After": str(retry)},
        )

    job_id = uuid.uuid4().hex[:12]
    now = _now_iso()

    with _lock:
        _prune_locked()
        _jobs[job_id] = {
            "job_id": job_id,
            "user_id": user_id,
            "status": "queued",
            "question": payload.question,
            "payload": payload.model_dump(),
            "created_at": now,
            "created_ts": time.time(),
            "started_at": None,
            "finished_at": None,
            "finished_ts": None,
            "result": None,
            "error": None,
        }
        _active_count += 1
        _persist()

    threading.Thread(target=_run_job, args=(job_id, payload.model_dump()), daemon=True).start()

    logger.info("Submitted background job %s (sims=%s, len=%d, user=%s)", job_id, payload.n_simulations, len(payload.question), user_id or "anon")
    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        question=payload.question,
        created_at=now,
    )


@router.get("/rag/query/job/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    user: Optional[Dict[str, Any]] = Depends(get_optional_user),
    api_user: Optional[Dict[str, Any]] = Depends(get_api_key_user),
):
    """Poll a background job's status and result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    # Ownership check: authenticated jobs require matching identity.
    owner = job.get("user_id")
    if owner:
        caller = (user or api_user)
        caller_id = caller.get("user_id") or caller.get("sub") if caller else None
        if caller_id != owner:
            raise HTTPException(status_code=403, detail="Not authorized to view this job")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        error=job.get("error"),
        result=job.get("result"),
    )
