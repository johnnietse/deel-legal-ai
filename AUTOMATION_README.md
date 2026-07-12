# Automated CSV Ingestion Setup

## Overview
This document describes the automated ingestion system for the `employment_cases_large.csv` dataset (1,255 Canadian employment law cases) into the Pinecone vector database with Elasticsearch BM25 hybrid search.

## Components

### 1. Core Scripts
| File | Purpose |
|------|---------|
| `rag_pipeline/csv_ingestion_wrapper.py` | Main wrapper with progress tracking, batch processing, quota-aware retry |
| `rag_pipeline/legal_document_ingester.py` | Core ingestion logic with `--csv-path` support |
| `rag_pipeline/embeddings.py` | Retry logic with exponential backoff (5 retries, 5-60s) |

### 2. Automation Options

#### Option A: Windows Task Scheduler (Local/VM)
```powershell
# Run setup script
.\setup_scheduled_task.ps1 -BatchSize 20 -MaxRetries 3

# Or with custom settings
.\setup_scheduled_task.ps1 -TaskName "OpenJustice-CSV-Ingestion" -BatchSize 10 -MaxRetries 5
```

**Task Details:**
- Runs every 30 minutes
- Processes 20 cases per run (configurable)
- 3 retries with exponential backoff on quota exhaustion
- Runs on battery, restarts on failure (3x, 5 min interval)
- 2-hour execution time limit

**Management Commands:**
```powershell
# View task
Get-ScheduledTask -TaskName "OpenJustice-CSV-Ingestion"

# Run manually
Start-ScheduledTask -TaskName "OpenJustice-CSV-Ingestion"

# View history
Get-ScheduledTaskInfo -TaskName "OpenJustice-CSV-Ingestion"

# Disable
Disable-ScheduledTask -TaskName "OpenJustice-CSV-Ingestion"

# Remove
Unregister-ScheduledTask -TaskName "OpenJustice-CSV-Ingestion" -Confirm:$false
```

#### Option B: GitHub Actions (Cloud)
File: `.github/workflows/csv-ingestion.yml`

**Features:**
- Runs every 30 minutes via cron (`*/30 * * * *`)
- Manual trigger via workflow_dispatch
- Configurable inputs (batch_size, max_retries, reset_progress)
- Uploads progress artifact
- Posts summary on completion

**Required Secrets:**
| Secret | Description |
|--------|-------------|
| `PINECONE_API_KEY` | Pinecone API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `GOOGLE_OAUTH_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth client secret |
| `JWT_SECRET_KEY` | JWT signing key |
| `ELASTICSEARCH_CLOUD_ID` | Elastic Cloud deployment ID |
| `ELASTICSEARCH_API_KEY` | Elastic Cloud API key |

**Manual Trigger:**
1. Go to Actions tab → "CSV Ingestion Automation"
2. Click "Run workflow"
3. Configure inputs (optional)
4. Click "Run workflow"

#### Option C: Manual Batch File (Windows)
```cmd
# Run with defaults (batch=20, retries=3)
run_ingestion.bat

# Custom batch size and retries
run_ingestion.bat 10 2

# Reset progress and run
run_ingestion.bat 20 3 reset
```

## Progress Tracking

### Progress File: `rag_pipeline/ingestion_progress.json`
```json
{
  "completed_case_ids": ["1", "2", "3", ...],
  "total_cases": 1255,
  "last_run": "2026-07-08T00:46:04.454772",
  "total_upserted": 143,
  "total_failed": 0
}
```

**Key Fields:**
- `completed_case_ids`: Array of Caseid strings already processed
- `total_upserted`: Total vectors successfully upserted to Pinecone
- `total_failed`: Cases that permanently failed (not retried)
- `last_run`: ISO timestamp of last successful batch

## Quota Management

### Gemini Free Tier Limits
- **100 requests/minute** for `gemini-embedding-001`
- Resets automatically (rolling window)

### Retry Strategy
| Attempt | Wait Time | Max Wait |
|---------|-----------|----------|
| 1 | 2-3s | 45s (from API) |
| 2 | 4-5s | 45s |
| 3 | 8-9s | 45s |
| 4 | 16-17s | 45s |
| 5 | 32-33s | 45s |

**Wrapper Behavior:**
1. On 429 error → waits per RetryInfo from API (or exponential backoff)
2. Max 5 retries per embedding (configurable via `--max-retries` for batch)
3. On batch failure after max retries → skips batch, logs failure, continues next run
4. Only marks cases "completed" on **successful** upsert

## Monitoring

### Local Monitoring
```powershell
# Check progress
Get-Content rag_pipeline/ingestion_progress.json | ConvertFrom-Json

# Watch Pinecone stats
python -c "
import os; os.environ['PINECONE_API_KEY']='...'
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index('deel-legal-cases')
print(index.describe_index_stats())
"
```

### GitHub Actions Monitoring
- View workflow runs: Actions tab → "CSV Ingestion Automation"
- Download progress artifact from completed runs
- Check workflow summary for upserted/failed counts

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Rate limited" errors persist | Wait for quota reset (check Gemini dashboard), increase `--max-retries` |
| Cases not marked completed | Check `ingestion_progress.json` - only successful upserts are recorded |
| Pinecone upsert fails | Check API key, index name, network connectivity |
| Task Scheduler doesn't run | Check "Run only if network available", verify Python path |
| GitHub Actions fails | Verify all secrets are set, check workflow logs |

### Reset Progress
```bash
# Via wrapper
python rag_pipeline/csv_ingestion_wrapper.py --reset

# Or manually
echo '{"completed_case_ids":[],"total_cases":0,"last_run":null,"total_upserted":0,"total_failed":0}' > rag_pipeline/ingestion_progress.json
```

## Current Status (as of 2026-07-08)

| Metric | Value |
|--------|-------|
| Pinecone vectors (legal_cases) | 738 |
| Pinecone vectors (legal_cases_docs) | 9 |
| CSV cases ingested | ~143 (of 1,255) |
| Remaining | ~1,112 |
| Elasticsearch BM25 | Configured |
| Hybrid search | Operational |

## Next Steps

1. **Wait for quota reset** (Gemini free tier: 100 req/min rolling window)
2. **Run test batch**: `python rag_pipeline/csv_ingestion_wrapper.py --batch-size 5 --max-retries 2`
2. **Deploy automation**: Choose Option A (Task Scheduler) or B (GitHub Actions)
3. **Monitor first few runs** to verify quota handling
4. **Scale batch size** once quota limits are understood (paid tier = higher limits)

## Files Summary

```
Law_AI_Deel/
├── .github/workflows/csv-ingestion.yml      # GitHub Actions workflow
├── setup_scheduled_task.ps1                 # Windows Task Scheduler setup
├── run_ingestion.bat                        # Manual run script
├── rag_pipeline/
│   ├── csv_ingestion_wrapper.py             # Main automation wrapper
│   ├── legal_document_ingester.py           # Core ingestion logic
│   ├── embeddings.py                        # Embedding with retry logic
│   └── ingestion_progress.json              # Progress tracking (auto-generated)
└── data/employment_cases_large.csv          # Source dataset (1,255 cases)
```