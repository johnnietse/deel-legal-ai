# OpenJustice.ai — Backup & Disaster Recovery Strategy

## Overview

This document defines the backup and disaster recovery strategy for OpenJustice.ai. The strategy covers all persistent data stores and provides RPO/RTO targets for each tier.

## RPO / RTO Targets

| Data Tier            | RPO (Recovery Point Obj.) | RTO (Recovery Time Obj.) | Priority |
|----------------------|---------------------------|--------------------------|----------|
| PostgreSQL (Auth)    | 5 minutes                 | 1 hour                   | Critical |
| Elasticsearch (BM25) | 1 hour                    | 4 hours                  | High     |
| Milvus (Vectors)     | 1 hour                    | 4 hours                  | High     |
| MinIO (Documents)    | 15 minutes                | 2 hours                  | High     |
| Redis (Cache)        | 1 hour (snapshot)         | 30 minutes               | Medium   |
| Application Config   | Per deployment            | 30 minutes               | Medium   |
| K8s Manifests        | Per commit                | 1 hour                   | Medium   |

## Backup Schedule

| Component       | Frequency        | Type            | Retention        | Storage Location     |
|-----------------|------------------|-----------------|------------------|----------------------|
| PostgreSQL      | Every 5 min (WAL) | Continuous WAL  | 30 days          | Azure Blob (backups) |
| PostgreSQL      | Daily            | Full snapshot   | 90 days          | Azure Blob (backups) |
| PostgreSQL      | Weekly           | Full + WAL      | 12 months        | Azure Blob (cold)    |
| Elasticsearch   | Daily            | Snapshot        | 30 days          | Azure Blob (backups) |
| Milvus          | Daily            | milvus-backup   | 30 days          | MinIO → Azure Blob   |
| MinIO (docs)    | Continuous       | S3 replication  | 30 days + GA     | Azure Blob (backups) |
| Redis           | Every 6 hours    | RDB snapshot    | 7 days           | Azure Blob (backups) |
| K8s Manifests   | Per commit       | Git-based       | Forever (git)    | GitHub + Azure Blob  |

## Backup Procedures

### 1. PostgreSQL Backup

PostgreSQL uses WAL archiving for continuous backup and `pg_dump` for logical snapshots.

```bash
# Full logical backup
PGPASSWORD=$POSTGRES_PASSWORD pg_dump \
  -h postgres-service \
  -U openjustice_admin \
  -d openjustice \
  -F c \
  -f /backups/postgres/openjustice_$(date +%Y%m%d_%H%M%S).dump

# WAL archiving (configured in postgresql.conf)
archive_command = 'az storage blob upload --container-name backups --name wal/%f --file %p --account-name $STORAGE_ACCOUNT'
```

### 2. Elasticsearch Snapshot

```bash
# Register snapshot repository
curl -X PUT "elasticsearch:9200/_snapshot/azure_backup" -H 'Content-Type: application/json' -d'
{
  "type": "azure",
  "settings": {
    "container": "backups",
    "base_path": "elasticsearch/",
    "chunk_size": "64mb",
    "compress": true
  }
}'

# Create snapshot
curl -X PUT "elasticsearch:9200/_snapshot/azure_backup/snapshot_$(date +%Y%m%d)"
```

### 3. Milvus Backup

Using [milvus-backup](https://github.com/zilliztech/milvus-backup) tool:

```bash
milvus-backup create -c default -n milvus_backup_$(date +%Y%m%d)
milvus-backup export -n milvus_backup_$(date +%Y%m%d) -o /backups/milvus/
az storage blob upload --container-name backups --name milvus/... --file ...
```

### 4. MinIO / Blob Storage Replication

MinIO data is continuously replicated to Azure Blob Storage using `mc mirror` or built-in bucket replication:

```bash
mc mirror --watch /data/minio azure/backups/minio/
```

## Disaster Recovery Runbook

### DR Tiers

| Tier   | Description                          | Max Downtime |
|--------|--------------------------------------|-------------|
| Tier 1 | Single pod/container failure         | Automatic   |
| Tier 2 | Node/VM failure in AKS               | < 5 minutes |
| Tier 3 | Azure region failure                 | < 4 hours   |
| Tier 4 | Data corruption / accidental delete  | < 4 hours   |
| Tier 5 | Full application disaster            | < 8 hours   |

### DR Plan — Tier 3 (Region Failure)

```
1. FAILOVER: Activate secondary region
   - az aks get-credentials --resource-group rg-openjustice-dr --name aks-openjustice-dr
   - kubectl apply -f k8s/

2. RESTORE POSTGRES:
   - az storage blob download --container backups --name postgres/latest.dump --file /tmp/restore.dump
   - pg_restore -h postgres-dr -U openjustice_admin -d openjustice /tmp/restore.dump

3. RESTORE ELASTICSEARCH:
   - Register snapshot repo pointing to backup container
   - Restore from latest snapshot

4. RESTORE MILVUS:
   - Download latest milvus-backup artifact
   - milvus-backup restore -n latest_backup

5. VERIFY:
   - kubectl rollout status deployment/legal-ai-api
   - curl -f https://api.openjustice.ai/health
   - Run integration smoke tests
```

### DR Plan — Tier 4 (Data Corruption)

```
1. IDENTIFY scope of corruption
2. STOP services writing to affected data store
3. RESTORE from latest clean backup:
   - Point-in-time recovery for PostgreSQL (15s granularity)
   - Snapshot restore for ES / Milvus
4. VERIFY data integrity
5. RESUME services
6. RUN reconciliation job if cross-store inconsistencies exist
```

## Automated Backup Script

The `backup.sh` script in `scripts/` automates all backups. It should be scheduled as a Kubernetes CronJob.

## Restore Script

The `restore.sh` script in `scripts/` automates restore operations. It accepts a timestamp parameter for point-in-time recovery.

## Testing

- **Daily**: Automated health check verifies backup files exist and are non-empty
- **Weekly**: Restore a backup to a test environment and verify data integrity
- **Monthly**: Full DR drill — simulate region failure and measure RTO

## Security

- Backups are encrypted at rest (Azure SSE) and in transit (TLS)
- Backup storage accounts have public access disabled
- Access to backups requires Azure RBAC (Contributor or Storage Blob Data Contributor)
- Backup files are purged automatically according to retention policy

## Contact

| Role              | Name              | Email                | Phone        |
|-------------------|-------------------|----------------------|--------------|
| DevOps Lead       | DevOps On-Call    | oncall@openjustice.ai| TBD          |
| DBA / Data Lead   | Data On-Call      | data@openjustice.ai  | TBD          |
| Security Lead     | Security On-Call  | security@openjustice.ai | TBD       |
