# Code Quality Standards

> Python coding standards for the Legal AI RAG System. Based on patterns found across the codebase.

## Import Conventions

**Always use `sys.path.insert(0, ...)` at the top of scripts** to resolve project-relative imports:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Prefer lazy imports** in `__init__.py` via `__getattr__` dispatch (see `rag_pipeline/__init__.py`). This prevents ImportErrors when optional dependencies (bs4, selenium, pinecone) are missing.

**Import config constants**, never hardcode paths:
```python
from config import CANLII_PDF_DOWNLOAD_DIR, LOG_FORMAT, LOG_LEVEL  # Good
EMBEDDING_MODEL = "text-embedding-004"  # Bad — should be in config.py
```

## Logging Pattern

Every module follows this exact pattern:
```python
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
```

**Log levels**:
- `logger.info()` for pipeline lifecycle (stages, counts, durations)
- `logger.warning()` for recoverable issues (missing data, fallbacks)
- `logger.error()` for failures that still allow continuation
- Never `print()` — use logger everywhere

## Retry Pattern

All external API calls use `tenacity` with exponential backoff:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def embed_text(self, text: str) -> EmbeddingResult:
    ...
```

## Error Handling

- **Catch specific exceptions**, not bare `except:`:
  ```python
  except requests.exceptions.RequestException as e:  # Good
  except Exception as e:  # Acceptable at pipeline boundaries
  ```
- **Log error, don't re-raise** in pipeline orchestration — let the pipeline continue
- **Use `try/except` around each storage operation** in `upsert_to_stores()` so one store failure doesn't block the other
- **Validate API keys** in `__init__` with `if not self.api_key: raise ValueError(...)`

## Dataclass Pattern

Use `@dataclass` for all response/value objects with type annotations:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class HybridResult:
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    bm25_score: float = 0.0
    vector_score: float = 0.0
    retrieved_by: List[str] = field(default_factory=list)
```

## Factory Pattern

Use factory functions with `backend` string param for backend abstraction:

```python
def create_vector_store(backend: str = "pinecone") -> VectorStore:
    if backend == "pinecone": return PineconeBackend(...)
    elif backend == "milvus": return MilvusBackend(...)
    raise ValueError(f"Unknown backend: {backend}")
```

References: `rag_pipeline/vector_store.py`, `rag_pipeline/search_engine.py`

## Config Management

- All tunable parameters in `config.py` — never hardcode in modules
- Use `os.getenv("VAR", "default")` with sensible defaults
- ByteDance sections are grouped with section comments like `# --- Hybrid Search Configuration (ByteDance §5.2) ---`
- Directory paths use `Path.mkdir(exist_ok=True)` in config initialization

## Codebase References

| Pattern | Location |
|---------|----------|
| Lazy imports via `__getattr__` | `rag_pipeline/__init__.py` |
| Factory pattern | `rag_pipeline/vector_store.py:create_vector_store()` |
| Dataclass with field defaults | `rag_pipeline/hybrid_retriever.py:HybridResult` |
| Retry decorator | `rag_pipeline/embeddings.py:GeminiEmbeddings.embed_text()` |
| Error isolation in pipeline | `rag_pipeline/pipeline.py:upsert_to_stores()` |
