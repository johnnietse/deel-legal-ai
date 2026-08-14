#!/usr/bin/env python
"""
Build Legal Knowledge Graph from Pinecone documents.

Extracts entities/relations from existing case documents and builds
the knowledge graph for GraphRAG retrieval.
"""
import sys
import os
sys.path.insert(0, r'C:\Users\Johnnie\Documents\Law_AI_Deel')
os.chdir(r'C:\Users\Johnnie\Documents\Law_AI_Deel')

from dotenv import load_dotenv
load_dotenv(r'C:\Users\Johnnie\Documents\Law_AI_Deel\.env')

from rag_pipeline.knowledge_graph import LegalKnowledgeGraph
from rag_pipeline.vector_store import create_vector_store
from config import PINECONE_INDEX_NAME, CHUNK_NAMESPACE, DOCUMENT_SUMMARY_NAMESPACE

def main():
    print("Building Legal Knowledge Graph...")
    
    # Initialize KG
    kg = LegalKnowledgeGraph()
    
    # Try to load existing
    try:
        kg.load()
        print(f"Loaded existing KG: {len(kg.graph.nodes())} nodes, {len(kg.graph.edges())} edges")
    except:
        print("No existing KG found, building from scratch")
    
    # Connect to vector store
    vs = create_vector_store(backend="pinecone", index_name=PINECONE_INDEX_NAME)
    vs.connect()
    
    # Fetch all documents from chunk namespace
    print("Fetching documents from Pinecone...")
    dummy_vector = [0.0] * 3072
    results = vs.search(
        query_vector=dummy_vector,
        top_k=10000,
        namespace=CHUNK_NAMESPACE,
        filter=None,
    )
    print(f"Fetched {len(results)} chunks")
    
    # Convert to document format for KG
    documents = []
    for r in results:
        documents.append({
            "id": r.id,
            "content": r.content,
            "metadata": r.metadata or {}
        })
    
    print(f"Building KG from {len(documents)} documents...")
    kg.build_from_documents(documents)
    
    # Save
    kg.save()
    print(f"KG saved: {len(kg.graph.nodes())} nodes, {len(kg.graph.edges())} edges")
    
    # Print some stats
    if hasattr(kg, 'graph') and kg.graph:
        import networkx as nx
        print(f"Graph density: {nx.density(kg.graph):.4f}")
        print(f"Connected components: {nx.number_weakly_connected_components(kg.graph)}")
        # Top PageRank nodes
        pr = nx.pagerank(kg.graph)
        top5 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 PageRank nodes:")
        for node, score in top5:
            print(f"  {node}: {score:.4f}")

if __name__ == "__main__":
    main()