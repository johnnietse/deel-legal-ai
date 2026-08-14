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

query = "worker classification Ontario"
embed_result = emb.embed_text(query)
if not embed_result.embedding:
    print("Failed to generate embedding")
    sys.exit(1)

results = vs.search(
    query_vector=embed_result.embedding,
    top_k=1,
    namespace=CHUNK_NAMESPACE,
    filter=None,
)
print(f'Fetched {len(results)} chunks')

for r in results:
    print(f'Content preview: {r.content[:200]}...')
    triples = kg.extract_triples_from_text(r.content, r.id)
    print(f'Extracted {len(triples)} triples:')
    for t in triples:
        print(f'  {t.subject} --{t.predicate}--> {t.object} ({t.subject_type}/{t.object_type}) conf={t.confidence}')