import sys, os
sys.path.insert(0, r'C:\Users\Johnnie\Documents\Law_AI_Deel')
os.chdir(r'C:\Users\Johnnie\Documents\Law_AI_Deel')

from dotenv import load_dotenv
load_dotenv(r'C:\Users\Johnnie\Documents\Law_AI_Deel\.env')

from rag_pipeline.vector_store import create_vector_store
from config import PINECONE_INDEX_NAME

vs = create_vector_store(backend='pinecone', index_name=PINECONE_INDEX_NAME)
vs.connect()
dummy = [0.0]*3072
for ns in ['legal_cases', 'legal_cases_docs', '']:
    results = vs.search(query_vector=dummy, top_k=5, namespace=ns, filter=None)
    print(f'Namespace "{ns}": {len(results)} chunks')