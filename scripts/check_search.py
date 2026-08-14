import sys, os
sys.path.insert(0, r'C:\Users\Johnnie\Documents\Law_AI_Deel')
os.chdir(r'C:\Users\Johnnie\Documents\Law_AI_Deel')

from dotenv import load_dotenv
load_dotenv(r'C:\Users\Johnnie\Documents\Law_AI_Deel\.env')

from rag_pipeline.embeddings import GeminiEmbeddings
from rag_pipeline.vector_store import create_vector_store
from config import PINECONE_INDEX_NAME, CHUNK_NAMESPACE

emb = GeminiEmbeddings()
vs = create_vector_store(backend='pinecone', index_name=PINECONE_INDEX_NAME)
vs.connect()

# Use a real query embedding
query = "worker classification Ontario"
embed_result = emb.embed_text(query)
if embed_result.embedding:
    results = vs.search(
        query_vector=embed_result.embedding,
        top_k=5,
        namespace=CHUNK_NAMESPACE,
        filter=None,
    )
    print(f'Namespace "legal_cases": {len(results)} chunks')
    for r in results[:3]:
        print(f'  {r.id}: {r.content[:80]}...')
else:
    print('Failed to generate embedding')