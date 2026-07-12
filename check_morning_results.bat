@echo off
cd /d "%~dp0"
echo ========================================
echo Morning Check — Overnight Embedder Results
echo ========================================
echo.
echo Last log entries:
echo ------------------------
findstr /r "^" data\background_embedder.log 2>nul | findstr /n "." | findstr /r "^[0-9]*:" | findstr "Embedded Upserted INGESTION COMPLETE"
echo.
echo Pinecone status:
python -c "import os; os.environ['PINECONE_API_KEY']='pcsk_5rfJUm_FvMtLsRyVxph343zYRaMaSe6QLXDRqPpRbqjP5jQAoaeDnsZjcitns4bMZUwm3Z'; from pinecone import Pinecone; pc=Pinecone(api_key=os.environ['PINECONE_API_KEY']); stats=pc.Index('deel-legal-cases').describe_index_stats(); print(f'Total vectors: {stats.total_vector_count}'); [print(f'  {ns}: {info.vector_count}') for ns,info in stats.namespaces.items()]"
echo.
echo Checkpoint progress:
if exist data\bg_embedder_checkpoint.json (
    python -c "import json; cp=json.load(open('data/bg_embedder_checkpoint.json')); print(f'  {len(cp.get(\"processed_ids\",[]))} chunks processed'); print(f'  Completed: {cp.get(\"completed\",False)}')"
) else (
    echo  No checkpoint found
)
echo.
echo ========================================
echo To test RAG: python -m pytest tests/ -x -v -k rag
echo ========================================
pause
