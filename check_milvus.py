from pymilvus import MilvusClient
client = MilvusClient("http://localhost:19530")
collections = client.list_collections()
print(f"Collections: {collections}")
for c in collections:
    stats = client.describe_collection(c)
    print(f"  {c}: rows={stats.get('row_count', '?')}")
