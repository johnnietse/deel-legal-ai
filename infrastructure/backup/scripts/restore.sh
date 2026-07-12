#!/bin/bash
# OpenJustice.ai — Automated Restore Script
# Usage: ./restore.sh [--postgres | --elasticsearch | --milvus | --redis] <backup_timestamp|latest>
# Example: ./restore.sh --postgres latest
#          ./restore.sh --postgres 20261225_000000
#          ./restore.sh --all latest

set -euo pipefail

# ================================
# Configuration
# ================================
BACKUP_ROOT="/backups"
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

# Redis
REDIS_HOST=${REDIS_HOST:-redis}
REDIS_PORT=${REDIS_PORT:-6379}

LOG_FILE="${BACKUP_ROOT}/logs/restore_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${BACKUP_ROOT}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error_exit() {
    log "ERROR: $*"
    exit 1
}

# ================================
# Utility Functions
# ================================
get_latest_backup() {
    local prefix="$1"
    local extension="$2"

    if command -v az &>/dev/null; then
        local latest
        latest=$(az storage blob list \
            --account-name "${STORAGE_ACCOUNT}" \
            --container-name "${STORAGE_CONTAINER}" \
            --prefix "${prefix}" \
            --query "max_by([], {lastModified: properties.lastModified}).name" \
            -o tsv 2>/dev/null) || true
        echo "$latest"
    else
        log "WARNING: Azure CLI not available, searching local filesystem"
        find "${BACKUP_ROOT}/${prefix}" -name "*.${extension}" -type f -printf '%T@ %p\n' \
            | sort -rn | head -1 | cut -d' ' -f2
    fi
}

download_from_blob() {
    local blob_path="$1"
    local dest_path="$2"

    log "Downloading azure://${STORAGE_CONTAINER}/${blob_path} to ${dest_path}"
    mkdir -p "$(dirname "${dest_path}")"

    az storage blob download \
        --account-name "${STORAGE_ACCOUNT}" \
        --container-name "${STORAGE_CONTAINER}" \
        --name "${blob_path}" \
        --file "${dest_path}" \
        --overwrite \
        2>&1 | tee -a "$LOG_FILE"
}

confirm_restore() {
    local component="$1"
    local backup_ref="$2"

    echo ""
    echo "⚠️  WARNING: You are about to restore ${component} from backup: ${backup_ref}"
    echo "   This will OVERWRITE existing data."
    echo ""
    read -r -p "Are you sure? Type 'yes' to continue: " confirmation
    if [ "${confirmation}" != "yes" ]; then
        log "Restore cancelled by user"
        exit 0
    fi
}

# ================================
# PostgreSQL Restore
# ================================
restore_postgres() {
    local backup_ref="$1"
    log "=== Starting PostgreSQL restore ==="

    if [ -z "${PG_PASSWORD}" ]; then
        error_exit "PG_PASSWORD not set"
    fi

    local dump_file

    if [ "${backup_ref}" = "latest" ]; then
        local latest_blob
        latest_blob=$(get_latest_backup "postgres/" "dump")
        [ -z "${latest_blob}" ] && error_exit "No PostgreSQL backups found"
        dump_file="${BACKUP_ROOT}/postgres/restore_latest.dump"
        download_from_blob "${latest_blob}" "${dump_file}"
    else
        dump_file="${BACKUP_ROOT}/postgres/restore_${backup_ref}.dump"
        local blob_path="postgres/$(echo ${backup_ref} | cut -c1-8)/${PG_DATABASE}_${backup_ref}.dump"
        download_from_blob "${blob_path}" "${dump_file}"
    fi

    confirm_restore "PostgreSQL" "${dump_file}"

    export PGPASSWORD="${PG_PASSWORD}"

    # Terminate existing connections
    psql -h "${PG_HOST}" -p "${PG_PORT}" -U "${PG_USER}" -d "postgres" \
        -c "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = '${PG_DATABASE}' AND pid <> pg_backend_pid();" \
        2>&1 | tee -a "$LOG_FILE"

    # Drop and recreate database
    psql -h "${PG_HOST}" -p "${PG_PORT}" -U "${PG_USER}" -d "postgres" \
        -c "DROP DATABASE IF EXISTS ${PG_DATABASE};" 2>&1 | tee -a "$LOG_FILE"
    psql -h "${PG_HOST}" -p "${PG_PORT}" -U "${PG_USER}" -d "postgres" \
        -c "CREATE DATABASE ${PG_DATABASE};" 2>&1 | tee -a "$LOG_FILE"

    # Restore from dump
    pg_restore \
        -h "${PG_HOST}" \
        -p "${PG_PORT}" \
        -U "${PG_USER}" \
        -d "${PG_DATABASE}" \
        -v \
        --jobs=4 \
        "${dump_file}" 2>&1 | tee -a "$LOG_FILE"

    unset PGPASSWORD
    log "PostgreSQL restore complete"
}

# ================================
# Elasticsearch Restore
# ================================
restore_elasticsearch() {
    local backup_ref="$1"
    log "=== Starting Elasticsearch restore ==="

    confirm_restore "Elasticsearch" "${backup_ref}"

    local snapshot_name

    if [ "${backup_ref}" = "latest" ]; then
        # Get latest snapshot name from Azure
        snapshot_name=$(curl -s "http://${ES_HOST}:${ES_PORT}/_snapshot/azure_backup/_all" \
            | python3 -c "import sys,json; snapshots=json.load(sys.stdin)['snapshots']; print(sorted(snapshots, key=lambda x: x['end_time'])[-1]['snapshot'])" 2>/dev/null) || \
            error_exit "Failed to determine latest ES snapshot"
    else
        snapshot_name="openjustice_${backup_ref}"
    fi

    # Close all indices (required for restore)
    curl -s -X POST "http://${ES_HOST}:${ES_PORT}/_all/_close" | tee -a "$LOG_FILE"

    # Restore snapshot
    local response
    response=$(curl -s -X POST "http://${ES_HOST}:${ES_PORT}/_snapshot/azure_backup/${snapshot_name}/_restore" \
        -H 'Content-Type: application/json' \
        -d '{
            "indices": "*",
            "ignore_unavailable": true,
            "include_global_state": false,
            "rename_pattern": "(.+)",
            "rename_replacement": "$1"
        }')

    log "Elasticsearch restore initiated: ${response}"

    # Monitor restore progress
    sleep 5
    curl -s "http://${ES_HOST}:${ES_PORT}/_cat/recovery?v" | tee -a "$LOG_FILE"
    log "Elasticsearch restore complete"
}

# ================================
# Milvus Restore
# ================================
restore_milvus() {
    local backup_ref="$1"
    log "=== Starting Milvus restore ==="

    if ! command -v milvus-backup &>/dev/null; then
        error_exit "milvus-backup CLI not found"
    fi

    confirm_restore "Milvus" "${backup_ref}"

    local backup_name
    if [ "${backup_ref}" = "latest" ]; then
        backup_name=$(milvus-backup list -h "${MILVUS_HOST}" -p "${MILVUS_PORT}" \
            | tail -1 | awk '{print $1}') || error_exit "No Milvus backups found"
    else
        backup_name="milvus_backup_${backup_ref}"
    fi

    milvus-backup restore \
        -h "${MILVUS_HOST}" \
        -p "${MILVUS_PORT}" \
        -n "${backup_name}" \
        2>&1 | tee -a "$LOG_FILE"

    log "Milvus restore complete"
}

# ================================
# Redis Restore
# ================================
restore_redis() {
    local backup_ref="$1"
    log "=== Starting Redis restore ==="

    confirm_restore "Redis" "${backup_ref}"

    local rdb_file

    if [ "${backup_ref}" = "latest" ]; then
        local latest_blob
        latest_blob=$(get_latest_backup "redis/" "rdb")
        [ -z "${latest_blob}" ] && error_exit "No Redis backups found"
        rdb_file="${BACKUP_ROOT}/redis/restore_latest.rdb"
        download_from_blob "${latest_blob}" "${rdb_file}"
    else
        rdb_file="${BACKUP_ROOT}/redis/restore_${backup_ref}.rdb"
        local blob_path="redis/$(echo ${backup_ref} | cut -c1-8)/dump_${backup_ref}.rdb"
        download_from_blob "${blob_path}" "${rdb_file}"
    fi

    # Copy RDB to Redis data directory and restart
    log "Stopping Redis, replacing RDB, and restarting..."
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" SHUTDOWN NOSAVE || true
    sleep 2

    # The RDB file needs to be placed where Redis can read it
    # This depends on your Redis configuration — adjust accordingly
    cp "${rdb_file}" /data/dump.rdb

    log "Redis RDB file replaced. Redis will load this file on next startup."
    log "Redis restore complete"
}

# ================================
# Main
# ================================
main() {
    log "==========================================="
    log "OpenJustice.ai Restore — $(date)"
    log "==========================================="

    if [ $# -lt 1 ]; then
        echo "Usage: $0 [--all | --postgres | --elasticsearch | --milvus | --redis] <backup_timestamp|latest>"
        echo ""
        echo "Examples:"
        echo "  $0 --postgres latest"
        echo "  $0 --postgres 20261225_000000"
        echo "  $0 --all latest"
        exit 1
    fi

    local mode="$1"
    local backup_ref="${2:-latest}"

    case "$mode" in
        --all)
            restore_postgres "${backup_ref}"
            restore_elasticsearch "${backup_ref}"
            restore_milvus "${backup_ref}"
            restore_redis "${backup_ref}"
            ;;
        --postgres)
            restore_postgres "${backup_ref}"
            ;;
        --elasticsearch)
            restore_elasticsearch "${backup_ref}"
            ;;
        --milvus)
            restore_milvus "${backup_ref}"
            ;;
        --redis)
            restore_redis "${backup_ref}"
            ;;
        *)
            echo "Unknown mode: ${mode}"
            echo "Usage: $0 [--all | --postgres | --elasticsearch | --milvus | --redis] <backup_timestamp|latest>"
            exit 1
            ;;
    esac

    log "==========================================="
    log "Restore completed successfully"
    log "==========================================="
}

main "$@"
