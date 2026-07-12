#!/bin/bash
# OpenJustice.ai — Automated Backup Script
# Usage: ./backup.sh [--all | --postgres | --elasticsearch | --milvus | --minio | --redis]
# Scheduled via K8s CronJob — runs all backups by default

set -euo pipefail

# ================================
# Configuration
# ================================
BACKUP_ROOT="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DATE=$(date +%Y%m%d)
RETENTION_DAYS=${RETENTION_DAYS:-30}
STORAGE_ACCOUNT=${STORAGE_ACCOUNT:-stopenjustice}
STORAGE_CONTAINER=${STORAGE_CONTAINER:-backups}

# PostgreSQL
PG_HOST=${PG_HOST:-postgres-service}
PG_PORT=${PG_PORT:-5432}
PG_USER=${PG_USER:-openjustice_admin}
PG_PASSWORD=${PG_PASSWORD:-}
PG_DATABASE=${PG_DATABASE:-openjustice}

# Elasticsearch
ES_HOST=${ES_HOST:-elasticsearch}
ES_PORT=${ES_PORT:-9200}

# Milvus
MILVUS_HOST=${MILVUS_HOST:-milvus-standalone}
MILVUS_PORT=${MILVUS_PORT:-19530}

# MinIO
MINIO_ALIAS=${MINIO_ALIAS:-minio}
MINIO_ENDPOINT=${MINIO_ENDPOINT:-http://milvus-minio:9000}
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-minioadmin}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-minioadmin}

# Azure Blob Storage
AZURE_STORAGE_ACCOUNT=${AZURE_STORAGE_ACCOUNT:-${STORAGE_ACCOUNT}}
AZURE_STORAGE_KEY=${AZURE_STORAGE_KEY:-}

# Logging
LOG_FILE="${BACKUP_ROOT}/logs/backup_${BACKUP_DATE}.log"
mkdir -p "${BACKUP_ROOT}/logs" "${BACKUP_ROOT}/postgres" "${BACKUP_ROOT}/redis"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $*"
    exit 1
}

# ================================
# Pre-flight Checks
# ================================
check_prerequisites() {
    log "Checking prerequisites..."
    command -v az >/dev/null 2>&1 || error_exit "Azure CLI not found"
    command -v pg_dump >/dev/null 2>&1 || log "WARNING: pg_dump not found"
    command -v curl >/dev/null 2>&1 || error_exit "curl not found"
    command -v mc >/dev/null 2>&1 || log "WARNING: mc (MinIO client) not found"
    log "Prerequisites check complete"
}

upload_to_blob() {
    local source_path="$1"
    local blob_path="$2"
    log "Uploading ${source_path} to azure://${STORAGE_CONTAINER}/${blob_path}"
    az storage blob upload \
        --account-name "${AZURE_STORAGE_ACCOUNT}" \
        --account-key "${AZURE_STORAGE_KEY}" \
        --container-name "${STORAGE_CONTAINER}" \
        --name "${blob_path}" \
        --file "${source_path}" \
        --overwrite \
        2>&1 | tee -a "$LOG_FILE"
}

cleanup_old_backups() {
    log "Cleaning up backups older than ${RETENTION_DAYS} days..."
    find "${BACKUP_ROOT}" -type f -name "*.dump" -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true
    find "${BACKUP_ROOT}" -type f -name "*.rdb" -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true
    log "Cleanup complete"
}

# ================================
# PostgreSQL Backup
# ================================
backup_postgres() {
    log "=== Starting PostgreSQL backup ==="

    if [ -z "${PG_PASSWORD}" ]; then
        log "WARNING: PG_PASSWORD not set. Skipping PostgreSQL backup."
        return 1
    fi

    local backup_file="${BACKUP_ROOT}/postgres/${PG_DATABASE}_${TIMESTAMP}.dump"
    local blob_path="postgres/${BACKUP_DATE}/${PG_DATABASE}_${TIMESTAMP}.dump"

    export PGPASSWORD="${PG_PASSWORD}"
    pg_dump \
        -h "${PG_HOST}" \
        -p "${PG_PORT}" \
        -U "${PG_USER}" \
        -d "${PG_DATABASE}" \
        -F c \
        -b \
        -v \
        -f "${backup_file}" 2>&1 | tee -a "$LOG_FILE"

    if [ $? -eq 0 ] && [ -f "${backup_file}" ]; then
        log "PostgreSQL dump created: ${backup_file} ($(du -h "${backup_file}" | cut -f1))"
        upload_to_blob "${backup_file}" "${blob_path}"
        log "PostgreSQL backup complete"
    else
        error_exit "PostgreSQL backup failed"
    fi

    unset PGPASSWORD
}

# ================================
# Elasticsearch Snapshot
# ================================
backup_elasticsearch() {
    log "=== Starting Elasticsearch snapshot ==="

    local snapshot_name="openjustice_${TIMESTAMP}"

    # Register snapshot repository (idempotent)
    curl -s -X PUT "http://${ES_HOST}:${ES_PORT}/_snapshot/azure_backup" \
        -H 'Content-Type: application/json' \
        -d "{
            \"type\": \"azure\",
            \"settings\": {
                \"container\": \"${STORAGE_CONTAINER}\",
                \"base_path\": \"elasticsearch/\",
                \"chunk_size\": \"64mb\",
                \"compress\": true
            }
        }" > /dev/null 2>&1 || log "WARNING: Snapshot repo may already exist"

    # Create snapshot
    local response
    response=$(curl -s -X PUT "http://${ES_HOST}:${ES_PORT}/_snapshot/azure_backup/${snapshot_name}?wait_for_completion=true" \
        -H 'Content-Type: application/json' \
        -d '{
            "indices": "_all",
            "ignore_unavailable": true,
            "include_global_state": true
        }')

    if echo "$response" | grep -q '"state":"SUCCESS"'; then
        log "Elasticsearch snapshot ${snapshot_name} completed successfully"
    else
        log "WARNING: Elasticsearch snapshot response: $response"
    fi
}

# ================================
# Milvus Backup
# ================================
backup_milvus() {
    log "=== Starting Milvus backup ==="

    if command -v milvus-backup &>/dev/null; then
        local backup_name="milvus_backup_${TIMESTAMP}"
        milvus-backup create \
            -h "${MILVUS_HOST}" \
            -p "${MILVUS_PORT}" \
            -n "${backup_name}" \
            2>&1 | tee -a "$LOG_FILE"

        log "Milvus backup ${backup_name} created"
    else
        log "WARNING: milvus-backup CLI not found. Skipping Milvus backup."
        log "Install from: https://github.com/zilliztech/milvus-backup"
    fi
}

# ================================
# MinIO Backup (mirror to Azure Blob)
# ================================
backup_minio() {
    log "=== Starting MinIO backup ==="

    if command -v mc &>/dev/null; then
        # Configure MinIO client
        mc alias set "${MINIO_ALIAS}" "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" \
            >/dev/null 2>&1 || log "WARNING: MinIO alias config failed"

        # Mirror MinIO buckets to Azure Blob
        mc mirror --watch \
            "${MINIO_ALIAS}/milvus-bucket" \
            "azure/${STORAGE_CONTAINER}/minio/" \
            2>&1 | tee -a "$LOG_FILE" &

        log "MinIO sync initiated (running in background)"
    else
        log "WARNING: mc (MinIO client) not found. Skipping MinIO backup."
    fi
}

# ================================
# Redis Backup (RDB snapshot copy)
# ================================
backup_redis() {
    log "=== Starting Redis backup ==="

    # Trigger a RDB save
    redis-cli -h redis -p 6379 SAVE 2>&1 | tee -a "$LOG_FILE" || \
        log "WARNING: Redis SAVE command failed"

    # Wait for RDB to be written
    sleep 5

    # Copy RDB file
    local rdb_source="/data/dump.rdb"
    local rdb_dest="${BACKUP_ROOT}/redis/dump_${TIMESTAMP}.rdb"

    if [ -f "${rdb_source}" ]; then
        cp "${rdb_source}" "${rdb_dest}"
        upload_to_blob "${rdb_dest}" "redis/${BACKUP_DATE}/dump_${TIMESTAMP}.rdb"
        log "Redis backup complete: ${rdb_dest}"
    else
        log "WARNING: Redis RDB file not found at ${rdb_source}"
    fi
}

# ================================
# Kubernetes Resources Backup
# ================================
backup_kubernetes() {
    log "=== Starting Kubernetes resources backup ==="

    local k8s_backup_dir="${BACKUP_ROOT}/kubernetes/${BACKUP_DATE}"
    mkdir -p "${k8s_backup_dir}"

    # Export current K8s resources
    kubectl get all --all-namespaces -o yaml > "${k8s_backup_dir}/all_resources.yaml" 2>/dev/null || \
        log "WARNING: K8s resource export failed"

    upload_to_blob "${k8s_backup_dir}/all_resources.yaml" "kubernetes/${BACKUP_DATE}/all_resources.yaml"
    log "Kubernetes resources backup complete"
}

# ================================
# Main
# ================================
main() {
    log "==========================================="
    log "OpenJustice.ai Backup — ${TIMESTAMP}"
    log "==========================================="

    check_prerequisites

    local mode="${1:---all}"

    case "$mode" in
        --all)
            backup_postgres
            backup_elasticsearch
            backup_milvus
            backup_minio
            backup_redis
            backup_kubernetes
            ;;
        --postgres)
            backup_postgres
            ;;
        --elasticsearch)
            backup_elasticsearch
            ;;
        --milvus)
            backup_milvus
            ;;
        --minio)
            backup_minio
            ;;
        --redis)
            backup_redis
            ;;
        --kubernetes)
            backup_kubernetes
            ;;
        *)
            echo "Usage: $0 [--all | --postgres | --elasticsearch | --milvus | --minio | --redis | --kubernetes]"
            exit 1
            ;;
    esac

    cleanup_old_backups

    log "==========================================="
    log "Backup completed successfully"
    log "==========================================="
}

main "$@"
