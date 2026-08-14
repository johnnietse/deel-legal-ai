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
    # Call the internal method with debug
    import json
    from rag_pipeline.embeddings import GeminiChat
    chat = GeminiChat()
    
    entity_types_str = ", ".join(kg.KG_ENTITY_TYPES if hasattr(kg, 'KG_ENTITY_TYPES') else [])
    relation_types_str = ", ".join(kg.KG_RELATION_TYPES if hasattr(kg, 'KG_RELATION_TYPES') else [])
    
    prompt = f"""Extract legal knowledge graph triples from the following legal text.

TEXT:
{r.content[:3000]}

ENTITY TYPES: Case, Court, Judge, LegalTest, Factor, Jurisdiction, Party, Statute
RELATION TYPES: cites, applies_test, involves_factor, decided_by, supports_classification, overrules, distinguishes, enacted_by, amends, interprets

Extract triples in this JSON format:
{{
    "triples": [
        {{
            "subject": "<entity name>",
            "subject_type": "<one of the entity types>",
            "predicate": "<one of the relation types>",
            "object": "<entity name>",
            "object_type": "<one of the entity types>",
            "confidence": <float 0.0-1.0>
        }}
    ]
}}

Rules:
1. Only use entity types from the provided list
2. Only use relation types from the provided list
3. Normalize entity names (e.g., "Sagaz test" not "the test from Sagaz")
4. Include confidence scores reflecting certainty of the extracted relation
5. Extract ALL meaningful legal relationships, not just the most obvious ones
6. For cases, use the standard citation format when available

Respond ONLY with the JSON object."""
    
    print('Calling LLM...')
    try:
        response = chat.generate(prompt, temperature=0.1, max_tokens=2048)
        print(f'Raw LLM Response length: {len(response)}')
        print(f'Raw LLM Response: {response}')
        
        # Parse like the method does
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        
        print(f'Cleaned length: {len(cleaned)}')
        print(f'Cleaned: {cleaned[:500]}...')
        
        data = json.loads(cleaned)
        print(f'Parsed JSON keys: {data.keys()}')
        print(f'Triples count: {len(data.get("triples", []))}')
        
        for t in data.get("triples", []):
            print(f'  Triple: {t}')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()