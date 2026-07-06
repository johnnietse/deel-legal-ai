# Development Workflow

## Overview
The Law AI Deel project follows a structured development workflow covering changes to the RAG pipeline, API, ML models, and infrastructure.

## Workflow Stages

### 1. Discovery
- Load context first (ContextScout) before any implementation
- Check existing patterns in `rag_pipeline/`, `api/`, `models/`, `k8s/`
- For external deps, use ExternalScout to fetch current docs
- Review `.tmp/external-context/` for previously fetched docs

### 2. Planning
- Break complex features into atomic subtasks via TaskManager
- Each subtask must have clear acceptance criteria
- Track dependencies between subtasks
- Use `.tmp/tasks/{feature}/` for task tracking

### 3. Implementation
- Write modular, functional code following project patterns
- Run `python -m pytest tests/ -v --timeout=60` after changes
- All 86 tests should pass (1 may skip if API key is revoked)
- For K8s changes, validate with `kubectl apply --dry-run=client`

### 4. Review
- Run full test suite before committing
- Check for deprecated API usage (ES body=, pymilvus ORM, etc.)
- Verify security hardening (non-root user, CORS, input validation)
- Review `.opencode/context/core/standards/code-quality.md`

### 5. Deployment
- Infrastructure: `docker compose up -d`
- K8s: `kubectl apply -f k8s/`
- Verify health: `curl http://localhost:8000/health`
- CI/CD: GitHub Actions runs tests on push/PR to main

## Quick Reference
```bash
# Run tests
python -m pytest tests/ -v --timeout=60

# Check Docker status
docker compose ps

# Check Git status
git status
git log --oneline -5
```
