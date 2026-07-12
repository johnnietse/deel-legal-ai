"""Test RAG with new A2AJ data."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(__file__))
os.environ["VECTOR_STORE_BACKEND"] = "pinecone"
os.environ["PYTHONIOENCODING"] = "utf-8"

from rag_pipeline.rag_query import LegalRAGQuery
from time import time

async def main():
    rag = LegalRAGQuery()
    
    # Query 1: Employee vs independent contractor
    print("=" * 60)
    print("QUERY 1: Employee vs Independent Contractor")
    print("=" * 60)
    t0 = time()
    result = await rag.query(
        "What are the key factors in determining if someone is an employee vs independent contractor in Canada?",
        top_k=8
    )
    elapsed = time() - t0
    print(f"Time: {elapsed:.1f}s")
    print(f"Answer: {result['answer'][:600]}")
    print(f"\nConfidence: {result.get('confidence_score', 'N/A')}")
    
    sources = result.get("sources", result.get("context", []))
    if sources:
        print(f"\nSources ({len(sources)}):")
        for i, s in enumerate(sources[:5]):
            title = s.get("title") or s.get("case_name") or "?"
            src = s.get("source") or s.get("court") or "?"
            print(f"  {i+1}. [{src}] {title[:80]}")
    
    # Query 2: Constructive dismissal
    print("\n" + "=" * 60)
    print("QUERY 2: What is constructive dismissal?")
    print("=" * 60)
    t0 = time()
    result2 = await rag.query(
        "What constitutes constructive dismissal and what remedies are available in Ontario?",
        top_k=8
    )
    elapsed = time() - t0
    print(f"Time: {elapsed:.1f}s")
    print(f"Answer: {result2['answer'][:600]}")
    
    sources2 = result2.get("sources", result2.get("context", []))
    if sources2:
        print(f"\nSources ({len(sources2)}):")
        for i, s in enumerate(sources2[:5]):
            title = s.get("title") or s.get("case_name") or "?"
            src = s.get("source") or s.get("court") or "?"
            print(f"  {i+1}. [{src}] {title[:80]}")
    
    # Query 3: Human rights discrimination at work
    print("\n" + "=" * 60)
    print("QUERY 3: Human rights discrimination in workplace")
    print("=" * 60)
    t0 = time()
    result3 = await rag.query(
        "What are the grounds for discrimination under Canadian human rights law in the workplace?",
        top_k=8
    )
    elapsed = time() - t0
    print(f"Time: {elapsed:.1f}s")
    print(f"Answer: {result3['answer'][:600]}")
    
    sources3 = result3.get("sources", result3.get("context", []))
    if sources3:
        print(f"\nSources ({len(sources3)}):")
        for i, s in enumerate(sources3[:5]):
            title = s.get("title") or s.get("case_name") or "?"
            src = s.get("source") or s.get("court") or "?"
            print(f"  {i+1}. [{src}] {title[:80]}")

if __name__ == "__main__":
    asyncio.run(main())
