# Verify Pinecone Index Content
import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load config
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "deel-legal-cases"

def verify_pinecone():
    print("=" * 60)
    print("VERIFYING PINECONE INDEX CONTENT")
    print("=" * 60)
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Get Stats
    stats = index.describe_index_stats()
    print(f"\n📊 Index Stats:")
    print(f"   Total Vectors: {stats.total_vector_count}")
    print(f"   Dimensions: {stats.dimension}")
    print(f"   Metric: cosine")  # Default
    
    # Since we can't 'list' all IDs easily in Pinecone without listing them by prefix or dummy query,
    # we'll do a dummy query to fetch a sample, but relying on stats is the best for total count.
    
    # To list titles, we really need the IDs. 
    # In our population script, we used readable IDs. 
    # Let's try to query for everything (vector of all 0.1s isn't great but works for fetching)
    # Better: use the stats or known IDs from our data if we had them.
    
    print("\n🔍 Sampling Documents (Retrieving Metadata):")
    # Query to get matches
    results = index.query(
        vector=[0.0] * 768,
        top_k=100,  # Try to get up to 100
        include_metadata=True
    )
    
    print(f"   Found {len(results['matches'])} sample matches via generic query")
    
    print("\n📄 Document Inventory (Sample):")
    categories = {}
    
    for i, match in enumerate(results['matches']):
        meta = match.metadata
        title = meta.get('title', 'Unknown Title')
        topic = meta.get('topic', 'Uncategorized')
        
        # Categorize
        categories[topic] = categories.get(topic, 0) + 1
        
        if i < 20:  # Print first 20 titles
            print(f"   {i+1}. [{topic}] {title[:60]}...")
            
    if len(results['matches']) > 20:
        print(f"   ... and {len(results['matches']) - 20} more in this sample.")

    print("\n📈 Topic Breakdown (from sample):")
    for topic, count in sorted(categories.items()):
        print(f"   - {topic}: {count}")

if __name__ == "__main__":
    verify_pinecone()
