import sys, os
sys.path.insert(0, r'C:\Users\Johnnie\Documents\Law_AI_Deel')
os.chdir(r'C:\Users\Johnnie\Documents\Law_AI_Deel')

from dotenv import load_dotenv
load_dotenv(r'C:\Users\Johnnie\Documents\Law_AI_Deel\.env')

from rag_pipeline.embeddings import GeminiChat

chat = GeminiChat()

prompt = """You are a legal knowledge extraction system. Output ONLY valid JSON.

TEXT:
Worker Classification: IT Consultants and Contractors

INDUSTRY CHARACTERISTICS:
The IT industry heavily relies on contractor arrangements.

ENTITY TYPES: Case, Court, Judge, LegalTest, Factor, Jurisdiction, Party, Statute
RELATION TYPES: cites, applies_test, involves_factor, decided_by, supports_classification, overrules, distinguishes, enacted_by, amends, interprets

Extract triples as JSON:
{
  "triples": [
    {"subject": "string", "subject_type": "string", "predicate": "string", "object": "string", "object_type": "string", "confidence": 0.0}
  ]
}

RULES:
- ONLY use entity types: Case, Court, Judge, LegalTest, Factor, Jurisdiction, Party, Statute
- ONLY use relation types: cites, applies_test, involves_factor, decided_by, supports_classification, overrules, distinguishes, enacted_by, amends, interprets
- Output ONLY the JSON object. NO text, NO markdown, NO reasoning.
- If no triples found, return {"triples": []}"""

print('Calling LLM...')
try:
    response = chat.generate(prompt, temperature=0.1, max_tokens=2048)
    print(f'Raw LLM Response length: {len(response)}')
    print(f'Raw LLM Response: {response}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()