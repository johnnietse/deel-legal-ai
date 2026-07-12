# OpenJustice.ai — Search, DeepSearch & API Platform Enhancement

**Date:** 2026-07-11
**Status:** Approved Design
**Author:** Development Agent

---

## 1. Overview

This spec covers adding web search, a NotebookLM-style DeepSearch tool, audio summaries, user-facing API key enforcement, auth hardening, and a dedicated search UI to the existing OpenJustice.ai platform — all using **entirely free, self-hosted components**.

### Scope (8 components)

| # | Component | Type | Status |
|---|-----------|------|--------|
| 1 | SearXNG web search | New Docker service + Python client | New |
| 2 | DeepSearch tool | New orchestration module | New |
| 3 | Audio summaries (edge-tts) | New endpoint + frontend | New |
| 4 | Search UI (frontend) | New route + page | New |
| 5 | API Key middleware | Fix existing mock → real | Fix |
| 6 | Auth on core endpoints | Add `get_current_user` guard | Fix |
| 7 | Redis integration | Wire into rate limiting/caching | Fix |
| 8 | Subscribe/usage mocks | Fix existing mocks → real | Fix |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                              │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────────┐ │
│  │ Chat UI  │ │  Search UI   │ │DeepSearch  │ │ Audio Player     │ │
│  │ (exists) │ │  (new)       │ │ Page (new) │ │ (new)            │ │
│  └────┬─────┘ └──────┬───────┘ └─────┬──────┘ └────────┬─────────┘ │
└───────┼──────────────┼───────────────┼─────────────────┼────────────┘
        │              │               │                  │
┌───────▼──────────────▼───────────────▼──────────────────▼────────────┐
│                     FastAPI Backend (api/)                            │
│                                                                       │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐   │
│  │ /rag/query   │  │ /rag/deepsearch  │  │ /api/chat/tts         │   │
│  │ (exists)     │  │ (new)            │  │ (new)                 │   │
│  └──────┬───────┘  └───────┬──────────┘  └──────────┬────────────┘   │
│         │                  │                         │                │
│  ┌──────▼──────────────────▼─────────────────────────▼────────────┐  │
│  │              API Key Middleware (new)                           │  │
│  │  Checks: JWT Bearer OR X-API-Key header on ALL /rag/, /classify│  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
        │            │                │                │
   ┌────▼───┐   ┌────▼────┐    ┌──────▼───────┐  ┌────▼────┐
   │Pinecone│   │Elastic  │    │  SearXNG     │  │   DB    │
   │3072-dim│   │Search   │    │  (Docker)    │  │Postgres │
   │(exists)│   │(exists) │    │  (NEW)       │  │(exists) │
   └────────┘   └─────────┘    └──────────────┘  └─────────┘
                                     │
                               DuckDuckGo
                               Google (via)
                               Bing (via)
```

---

## 3. Component Details

### 3.1 SearXNG Web Search (NEW)

**Purpose:** Provide free, self-hosted web search capability for the DeepSearch tool.

**Implementation:**
- Add SearXNG service to `docker-compose.yml` (port 8888)
- Allow outgoing HTTP (SearXNG needs internet to query upstream engines)
- Create `rag_pipeline/web_search.py` with:
  - `SearxngClient` — wraps SearXNG JSON API (`/search?format=json`)
  - `search_web(query, top_k=5)` — returns ranked results with title, snippet, URL, content
  - Configurable engines (default: duckduckgo, google, bing, wikipedia)
  - Timeout: 10s per request
  - Error handling: if SearXNG is unreachable, gracefully degrade (skip web results)

**Configuration (in config.py):**
```python
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")
SEARXNG_ENABLED = True
```

**Docker addition:**
```yaml
searxng:
  image: searxng/searxng:latest
  container_name: searxng
  ports:
    - "8888:8080"
  volumes:
    - ./searxng:/etc/searxng:rw
  environment:
    - SEARXNG_BASE_URL=http://localhost:8888
    - SEARXNG_SECRET_KEY=${SEARXNG_SECRET_KEY:-change_me_in_production}
  cap_add:
    - NET_BIND_SERVICE
  restart: unless-stopped
```

**Required from user:** Nothing. Runs self-hosted in Docker.

---

### 3.2 DeepSearch Tool (NEW)

**Purpose:** NotebookLM-style multi-source deep research that searches case law, web, statutes, and BM25 simultaneously, then synthesizes a cited answer with follow-up capability.

**Implementation:** `rag_pipeline/deep_search.py`

**Orchestrator Flow:**
```
deep_search(query, user_id) -> DeepSearchResult
  1. Parse query → identify jurisdictions, topics, keywords
  2. Launch parallel searches:
     a. Pinecone (vector) → similar case law chunks
     b. Elasticsearch (BM25) → keyword-matched case law
     c. SearXNG (web) → current legal articles, news, updates
     d. PostgreSQL (legislation) → stored statutes
  3. Fuse all results with source-type tagging
  4. Rank by relevance (RRF fusion across all sources)
  5. Gemini synthesizes answer with [Source N] citations
  6. Tag each source: 📜 Case Law / 📖 Statute / 🌐 Web
  7. Return: answer, sources, source_type_counts, suggested_follow_ups
```

**Data model:**
```python
class DeepSearchResult(BaseModel):
    answer: str
    sources: List[UnifiedSource]  # merged, ranked
    source_type_counts: dict  # {"case_law": 3, "web": 2, "statute": 1}
    suggested_follow_ups: List[str]
    processing_time_ms: int

class UnifiedSource(BaseModel):
    id: str
    title: str
    excerpt: str
    url: str = ""
    source_type: Literal["case_law", "web", "statute", "bm25"]
    relevance_score: float
```

**Endpoint:** `POST /rag/deepsearch`
- Request: `{ "query": str, "max_sources": int (default 15) }`
- Response: `DeepSearchResult`
- Auth: Requires valid API key or JWT

**Frontend page:** `/deepsearch`
- Large query input (like NotebookLM's "build a briefing")
- Shows sources loading in real-time as they arrive
- Rendered answer with clickable source citations
- "Ask follow-up" input below the answer
- "Generate Audio Summary" button (triggers TTS)

---

### 3.3 Audio Summaries via edge-tts (NEW)

**Purpose:** Generate spoken summaries of DeepSearch results — NotebookLM's most popular feature, entirely free.

**Implementation:**
- `pip install edge-tts` (uses Microsoft's free TTS endpoint, no API key needed)
- Create `rag_pipeline/tts.py` with:
  - `generate_audio(text, voice="en-CA-LiamNeural")` → MP3 bytes
  - Canadian English voice by default (appropriate for Canadian law)
  - Streaming support for long responses
  - Caching: same text hash → skip regeneration

**Endpoint:** `POST /api/chat/tts`
- Request: `{ "text": str, "voice": str (optional) }`
- Response: Audio/MP3 stream
- Auth: Requires valid API key or JWT

**Frontend integration:**
- "🔊 Listen" button on DeepSearch results
- Audio player component (native `<audio>` element)
- Controls: play/pause, download, speed (1x, 1.5x, 2x)

**Available free voices (Microsoft):**
- `en-CA-LiamNeural` (Canadian male) — DEFAULT for legal content
- `en-CA-ClaraNeural` (Canadian female)
- `en-US-GuyNeural` (US male)

---

### 3.4 Search UI (NEW)

**Purpose:** Dedicated search page in the frontend with filters — distinct from the chat interface.

**Implementation:**
- New route: `/search` in `App.tsx`
- New page: `app/search/SearchPage.tsx`
- API endpoint: `POST /rag/search` (different from `/rag/query` — returns structured results, not a chat answer)

**Search page layout:**
```
┌─────────────────────────────────────────────────┐
│ 🔍 [________________________________] [Search] │
│ Filters: [Jurisdiction ▼] [Court ▼] [Date ▼]   │
├─────────────────────────────────────────────────┤
│ Results 1-10 of 143 (0.4s)                       │
│                                                  │
│ ┌─────────────────────────────────────────────┐ │
│ │ Smith v. Jones, 2024 ONSC 1234  ·  96%     │ │
│ │ 📜 Case Law · Ontario Superior Court       │ │
│ │ "The test for constructive dismissal..."    │ │
│ │ [View] [Cite]                              │ │
│ ├─────────────────────────────────────────────┤ │
│ │ Ontario ESA Amendments 2025  ·  88%        │ │
│ │ 🌐 Web · canlii.org                       │ │
│ │ "Recent changes to the Employment..."      │ │
│ │ [View] [Cite]                              │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

**Filters (frontend-only, applied as query params):**
- Jurisdiction dropdown (Ontario, Federal, British Columbia, etc.)
- Source type checkboxes (Case Law, Web, Statutes)
- Date range (from/to)
- Sort by (Relevance, Date)

---

### 3.5 API Key Middleware (FIX MOCK)

**Purpose:** Replace mock frontend data with real backend-backed API key management.

**What exists:**
- Backend: `ApiKey` model in `db/models.py` ✅
- Backend: CRUD in `db/repository.py` (`create_api_key`, `list_api_keys`, `delete_api_key`) ✅
- Frontend: Settings page UI (`ApiKeysTab`) ✅
- Frontend: `realClient.ts` methods (`getApiKeys`, `createApiKey`, `revokeApiKey`) ❌ RETURN MOCK DATA

**What needs to be built:**
1. **Backend API key endpoints** (in `api/auth.py` or new `api/api_keys.py`):
   - `GET /api/auth/keys` — List user's API keys (returns masked: `oj_...abc`)
   - `POST /api/auth/keys` — Generate new key (returns full key once)
   - `DELETE /api/auth/keys/{id}` — Revoke key
2. **Backend key generation:**
   - Prefix: `oj_` (OpenJustice)
   - Suffix: 48 chars of `secrets.token_urlsafe`
   - Hash stored in DB (SHA-256), plaintext returned once
3. **API Key middleware** (in `api/auth.py`):
   - New dependency `get_api_key_user(api_key: str = Header(None, alias="X-API-Key"))`
   - Hash the provided key → lookup in `api_keys` table → get `user_id` → return user
4. **Update `get_current_user`** to accept EITHER JWT Bearer OR `X-API-Key`
5. **Wire frontend** to real endpoints → remove mock data

**Key format:** `oj_living7843sunset+bat+US1A2b3C4d5E6f7G8h9I0jK...` (prefix + 48 random chars)

---

### 3.6 Auth on Core Endpoints (FIX)

**Purpose:** Currently `/rag/query`, `/classify`, `/rag/deepsearch`, `/rag/verify`, `/evaluate/*` have NO auth — anyone can call them.

**Implementation:**
- Make `get_current_user` optional on these endpoints (use `get_optional_user`)
- Track `queries_used` for authenticated users
- For unauthenticated requests: apply stricter rate limiting (IP-based, 5/min)
- For API key requests: track usage per key + per user

**Changes:**
- `api/main.py`: Add `user = await get_optional_user(request)` to `/rag/query`, `/classify`, `/rag/deepsearch`
- `rag_query.py`: Pass `user_id` through to metrics tracking
- Rate limit middleware: Apply to ALL paths (currently only `/api/`)

**Note:** Existing chat frontend already sends JWT tokens. This just enforces what the frontend already does.

---

### 3.7 Redis Integration (FIX)

**Purpose:** Redis is running in docker-compose.dev.yml but unused. Use it for:
1. **SearXNG result caching** (TTL: 1 hour)
2. **Rate limiting** (replace in-memory `RateStore` for persistence across restarts)
3. **Response caching** (optional, replaces in-memory TTL cache)

**Implementation:**
- Add `redis-py` to `requirements.txt`
- Create `rag_pipeline/redis_client.py`:
  - Lazy Redis connection (connect on first use)
  - `get(key)`, `set(key, value, ttl)`, `delete(key)`
  - Graceful fallback if Redis unavailable (log warning, skip cache)
- Update `RateStore` to optionally use Redis
- Update SearXNG client to cache results keyed by query hash

**Configuration:**
```python
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ENABLED = True
```

---

### 3.8 Fix Frontend Subscribe/Usage Mocks (FIX)

**What's mock:**
- `getUsageChartData()` — random numbers
- `getRecentActivity()` — hardcoded entries
- `upgradeSubscription()` — no-op

**Fix plan:**
- `getUsageChartData(days)` → Backend endpoint `GET /api/users/me/usage/chart?days=30` that returns real historical usage from `queries_used` tracking
- `getRecentActivity()` → Backend endpoint `GET /api/users/me/activity` that returns recent conversations + document uploads
- `upgradeSubscription()` → Initially returns mock (no payment processor). Add a `POST /api/subscriptions/upgrade` that records intent + updates tier. Payment integration would be a future step.

---

## 4. New Files Summary

| File | Purpose |
|------|---------|
| `rag_pipeline/web_search.py` | SearXNG client wrapper |
| `rag_pipeline/deep_search.py` | DeepSearch orchestrator |
| `rag_pipeline/tts.py` | edge-tts audio generation |
| `rag_pipeline/redis_client.py` | Redis connection + caching |
| `api/api_keys.py` | API key CRUD endpoints |
| `api/deepsearch.py` | DeepSearch + TTS endpoints |
| `openjustice-frontend/src/app/search/SearchPage.tsx` | Search UI page |
| `openjustice-frontend/src/app/deepsearch/DeepSearchPage.tsx` | DeepSearch page |
| `openjustice-frontend/src/components/search/SearchResultCard.tsx` | Result card component |
| `openjustice-frontend/src/components/search/SearchFilters.tsx` | Filter sidebar |
| `openjustice-frontend/src/components/search/AudioPlayer.tsx` | Audio playback component |
| `searxng/settings.yml` | SearXNG configuration |

## 5. Modified Files Summary

| File | Change |
|------|--------|
| `docker-compose.yml` | Add SearXNG service |
| `config.py` | Add SEARXNG_BASE_URL, REDIS_URL |
| `requirements.txt` | Add edge-tts, redis-py |
| `api/main.py` | Add auth to core endpoints, new router mounts |
| `api/router.py` | Include api_keys, deepsearch routers |
| `api/auth.py` | Add API key auth dependency |
| `api/middleware.py` | Extend to cover /rag/, /classify/ paths; Redis-backed store |
| `db/repository.py` | Add key generation + hash storage |
| `rag_pipeline/rag_query.py` | Accept user_id for tracking |
| `openjustice-frontend/src/App.tsx` | Add /search, /deepsearch routes |
| `openjustice-frontend/src/lib/api/realClient.ts` | Replace all mock data with real API calls |
| `openjustice-frontend/src/lib/hooks/useQuery.ts` | Add deepSearch, search hooks |
| `openjustice-frontend/src/app/settings/SettingsPage.tsx` | Connect to real API key endpoints |

## 6. Data Flow for Key Scenarios

### Scenario A: DeepSearch Query
```
1. User types "What are notice requirements for constructive dismissal in Ontario?"
2. POST /rag/deepsearch { query: "...", max_sources: 15 }
3. API Key middleware validates JWT or X-API-Key
4. DeepSearch launches 4 parallel fetches:
   a. Pinecone.search(query, top_k=10) → 10 case law chunks
   b. ES.search(query, top_k=10) → 10 BM25 results
   c. SearXNG.search(query, top_k=5) → 5 web results
   d. PostgreSQL: SELECT * FROM legislation WHERE content ILIKE '%constructive dismissal%'
5. All results merged → RRF ranked → top 15 selected
6. Gemini builds answer with [Source 1..15] citations
7. Response returned with unified source list + type tags
8. User clicks "🔊 Listen" → POST /api/chat/tts → MP3 audio
```

### Scenario B: API Key Registration
```
1. User logs in via JWT (existing flow)
2. Navigates to Settings → API Keys tab
3. Clicks "Create New Key" → POST /api/auth/keys
4. Backend: generates `oj_<48 random chars>`, stores SHA-256 hash
5. Returns full key once → frontend displays in yellow warning box
6. User copies key and uses in external app:
   curl -H "X-API-Key: oj_living7843sunset..." http://localhost:8000/rag/query -d '{"query":"..."}'
7. Backend: hashes incoming key → looks up hash in api_keys table → gets user_id → processes → tracks usage
```

### Scenario C: Audio Summary
```
1. DeepSearch result displayed
2. User clicks "🔊 Listen" on result
3. Frontend sends POST /api/chat/tts { text: answer_text }
4. Backend: edge-tts generates MP3 from text
5. Cache: same text hash → skip regeneration
6. Streams MP3 bytes back
7. Frontend <audio> player renders controls
```

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| SearXNG container not running | DeepSearch continues without web results, logs warning |
| SearXNG returns 0 results | DeepSearch uses vector + BM25 + legislation only |
| edge-tts fails | Return 503 with "Audio generation unavailable" |
| Invalid API key | Return 401 with "Invalid API key" |
| Expired JWT | Return 401 with "Token expired" — client refreshes via /api/auth/token |
| Redis unavailable | Rate limiting falls back to in-memory store |
| Gemini 429 (rate limit) | Key rotation handles (12 keys available) |
| All 12 Gemini keys exhausted | Return 429 with Retry-After header |

## 8. Testing

Each component gets:
- **Backend unit tests** (pytest) — mock external services
- **Integration test** — verifies SearXNG Docker container responds
- **Frontend component test** — verifies UI renders with mock data
- **E2E test** — full flow: search → DeepSearch → audio

## 9. Order of Implementation

Implementation should follow dependency order:

```
Phase 1: Infrastructure
  ├─ SearXNG Docker setup
  ├─ Redis integration
  └─ requirements.txt updates

Phase 2: Backend Fixes
  ├─ API Key endpoints + middleware
  ├─ Auth on core endpoints
  ├─ Subscribe/usage mocks → real
  └─ Rate limiting extended

Phase 3: New Backend Features
  ├─ web_search.py (SearXNG client)
  ├─ deep_search.py (orchestrator)
  └─ tts.py (audio)

Phase 4: Frontend
  ├─ Search UI page
  ├─ DeepSearch page
  ├─ Audio player
  └─ Fix all mock data in realClient.ts
```

---

## 10. Cost Verification

| Service | Cost | Notes |
|---------|------|-------|
| SearXNG | $0 | Self-hosted Docker, no API key |
| edge-tts | $0 | Free Microsoft endpoint |
| Gemini API (12 keys) | $0 | Free tier, separate projects |
| Pinecone | $0 | Free tier (100K vectors) |
| Redis | $0 | Docker, already in compose |
| Elasticsearch | $0 | Docker, already running |
| Milvus | $0 | Docker, already running |
| Neon PostgreSQL | $0 | Free tier |
| **Total** | **$0/month** | |

**No API keys needed from the user.** Everything runs on existing infrastructure.
