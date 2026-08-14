import sys, os
sys.path.insert(0, r'C:\Users\Johnnie\Documents\Law_AI_Deel')
os.chdir(r'C:\Users\Johnnie\Documents\Law_AI_Deel')

from dotenv import load_dotenv
load_dotenv(r'C:\Users\Johnnie\Documents\Law_AI_Deel\.env')

from rag_pipeline.knowledge_graph import LegalKnowledgeGraph
from rag_pipeline.vector_store import create_vector_store
from rag_pipeline.embeddings import GeminiEmbeddings
from config import PINECONE_INDEX_NAME, CHUNK_NAMESPACE

kg = LegalKnowledgeGraph()
vs = create_vector_store(backend='pinecone', index_name=PINECONE_INDEX_NAME)
vs.connect()
emb = GeminiEmbeddings()

# Use a real query embedding to fetch documents
query = "worker classification Ontario"
embed_result = emb.embed_text(query)
if not embed_result.embedding:
    print("Failed to generate embedding")
    sys.exit(1)

results = vs.search(
    query_vector=embed_result.embedding,
    top_k=10,
    namespace=CHUNK_NAMESPACE,
    filter=None,
)
print(f'Fetched {len(results)} chunks')

documents = []
for r in results:
    documents.append({
        "id": r.id,
        "content": r.content,
        "metadata": r.metadata or {}
    })

print(f'Building KG from {len(documents)} documents...')
kg.build_from_documents(documents)
kg.save()
print(f'KG saved: {len(kg.graph.nodes())} nodes, {len(kg.graph.edges())} edges')

if hasattr(kg, 'graph') and kg.graph:
    import networkx as nx
    print(f'Graph density: {nx.density(kg.graph):.4f}')
    print(f'Connected components: {nx.number_weakly_connected_components(kg.graph)}')
    pr = nx.pagerank(kg.graph)
    top5 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
    print('Top 5 PageRank nodes:')
    for node, score in top5:
        print(f'  {node}: {score:.4f}')