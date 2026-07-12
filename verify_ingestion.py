"""
Verify what's been ingested — quick status report.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Pinecone status
os.environ["PINECONE_API_KEY"] = "pcsk_5rfJUm_FvMtLsRyVxph343zYRaMaSe6QLXDRqPpRbqjP5jQAoaeDnsZjcitns4bMZUwm3Z"
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
idx = pc.Index("deel-legal-cases")
stats = idx.describe_index_stats()
print("=" * 60)
print("PINEQONE VECTOR STATUS")
print("=" * 60)
print(f"Total vectors: {stats.total_vector_count}")
for ns, info in stats.namespaces.items():
    print(f"  {ns}: {info.vector_count}")

# Check source distribution with sampling
import random
print()
print("Sample vectors by source:")
sample_vec = [random.uniform(-0.1, 0.1) for _ in range(3072)]
results = idx.query(vector=sample_vec, top_k=10, include_metadata=True, namespace="legal_cases")
for r in results.matches:
    meta = r.metadata or {}
    src = meta.get("source", "?")
    title = (meta.get("title") or meta.get("case_name") or "?")[:60]
    print(f"  [{src:20s}] {title}")

# A2AJ parquet files
import pandas as pd
data_dir = os.path.join(os.path.dirname(__file__), "data")
print()
print("=" * 60)
print("A2AJ DATASETS DOWNLOADED")
print("=" * 60)
a2aj_files = sorted(f for f in os.listdir(data_dir) if f.startswith("a2aj_") and f.endswith(".parquet"))
total_mb = 0
total_cases = 0
for f in a2aj_files:
    sz = os.path.getsize(os.path.join(data_dir, f))
    mb = sz / (1024 * 1024)
    total_mb += mb
    df = pd.read_parquet(os.path.join(data_dir, f))
    ds_name = f.replace("a2aj_", "").replace("_employment.parquet", "")
    total_cases += len(df)
    print(f"  {ds_name:8s} {len(df):6d} cases  ({mb:6.1f} MB)")

print(f"  {'TOTAL':8s} {total_cases:6d} cases  ({total_mb:6.1f} MB)")

# Check if background embedder is running
print()
print("=" * 60)
print("BACKGROUND EMBEDDER STATUS")
print("=" * 60)
cp_file = os.path.join(data_dir, "bg_embedder_checkpoint.json")
if os.path.exists(cp_file):
    import json
    with open(cp_file) as f:
        cp = json.load(f)
    processed = len(cp.get("processed_ids", []))
    print(f"  Processed: {processed} chunks")
    if cp.get("completed"):
        print(f"  Status: COMPLETED")
    else:
        print(f"  Status: IN PROGRESS (or PAUSED)")

log_file = os.path.join(data_dir, "background_embedder.log")
if os.path.exists(log_file):
    log_size = os.path.getsize(log_file)
    print(f"  Log size: {log_size / 1024:.0f} KB")

# Check other services
print()
print("=" * 60)
print("SYSTEM STATUS")
print("=" * 60)
# Check Elasticsearch
try:
    import requests
    es_resp = requests.get("http://localhost:9200/", timeout=5)
    if es_resp.status_code == 200:
        es_ver = es_resp.json().get("version", {}).get("number", "?")
        print(f"  Elasticsearch: RUNNING (v{es_ver})")
    else:
        print(f"  Elasticsearch: ERROR ({es_resp.status_code})")
except Exception as e:
    print(f"  Elasticsearch: DOWN ({e})")

# Check Milvus
try:
    from pymilvus import connections, utility
    connections.connect(host="localhost", port="19530")
    if utility.has_collection("deel_legal_cases"):
        print(f"  Milvus: RUNNING (collection deel_legal_cases exists)")
    else:
        print(f"  Milvus: RUNNING (no collection)")
    connections.disconnect("default")
except Exception as e:
    print(f"  Milvus: DOWN ({e})")

# Check Pinecone free tier usage
print()
print("=" * 60)
print("INFERENCE")
print("=" * 60)
free_tier_limit = 100000
pct = stats.total_vector_count / free_tier_limit * 100
print(f"  Pinecone free tier: {stats.total_vector_count}/{free_tier_limit} ({pct:.1f}%)")
print(f"  Remaining capacity: {free_tier_limit - stats.total_vector_count:,} vectors")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Total vectors in DB: {stats.total_vector_count}")
print(f"  A2AJ cases ready for embedding: {total_cases}")
print(f"  Already embedded from A2AJ: ~{stats.total_vector_count - 1343}")

# Estimate remaining
remaining = total_cases - (stats.total_vector_count - 1343)
embed_rate_per_hour = 10 * 60 / 75  # ~8 per min = ~480/hr
print(f"  Remaining to embed: ~{remaining}")
print(f"  Est. time at current rate: ~{remaining / 8 / 60:.1f} hours")
print(f"  Tip: Use --upsert with smaller batches for faster completion")
