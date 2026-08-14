import sys, os
sys.path.insert(0, r'C:\Users\Johnnie\Documents\Law_AI_Deel')
os.chdir(r'C:\Users\Johnnie\Documents\Law_AI_Deel')

from dotenv import load_dotenv
load_dotenv(r'C:\Users\Johnnie\Documents\Law_AI_Deel\.env')

from rag_pipeline.embeddings import GeminiChat

chat = GeminiChat()

prompt = "Say 'hello' in JSON format: {\"message\": \"hello\"}"

print('Calling LLM...')
try:
    response = chat.generate(prompt, temperature=0.1, max_tokens=50)
    print(f'Response: {response}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()