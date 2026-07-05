# RAG Pipeline - Hybrid Graph + Vector Retriever
"""
Hybrid retriever combining vector similarity search with knowledge graph traversal.

Instead of relying solely on vector similarity (which retrieves isolated chunks),
this module merges results from both:
1. Pinecone vector search (semantic similarity)
2. Knowledge graph traversal (structured relationships)

The fusion approach provides both:
- Relevant text passages (from vector search)
- Structured relationship context (from graph traversal)
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import GeminiEmbeddings, GeminiChat
from rag_pipeline.vector_store import create_vector_store, VectorSearchResult
from rag_pipeline.knowledge_graph import LegalKnowledgeGraph, SubgraphResult

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid vector + graph retrieval"""
    query: str
    vector_results: List[Dict[str, Any]]
    graph_results: List[SubgraphResult]
    merged_context: str
    answer: str
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """
    Combines vector similarity search with knowledge graph traversal.
    
    Retrieval flow:
    1. Vector search via Pinecone for relevant text passages
    2. Extract entities from query + retrieved passages
    3. Traverse knowledge graph from those entities
    4. Merge vector results + graph context into unified context
    5. Generate answer with enriched context
    
    The graph context adds structured relationship information that
    pure vector search cannot capture — precedent chains, factor
    relationships, and legal test applications.
    """
    
    def __init__(
        self,
        embeddings: Optional[GeminiEmbeddings] = None,
        chat: Optional[GeminiChat] = None,
        vector_store=None,
        knowledge_graph: Optional[LegalKnowledgeGraph] = None,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.embeddings = embeddings or GeminiEmbeddings()
        self.chat = chat or GeminiChat()
        self.vector_store = vector_store if vector_store is not None else create_vector_store()
        self.kg = knowledge_graph or LegalKnowledgeGraph()
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

        # Ensure vector store is connected
        self.vector_store.connect()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        graph_depth: int = 2,
    ) -> HybridResult:
        """
        Perform hybrid retrieval combining vector search and graph traversal.
        
        Args:
            query: Legal question to answer
            top_k: Number of vector search results
            namespace: Pinecone namespace
            filter: Metadata filter for vector search
            graph_depth: BFS depth for graph traversal
            
        Returns:
            HybridResult with merged context and generated answer
        """
        logger.info(f"Hybrid retrieval for: {query[:60]}...")
        
        # Step 1: Vector search
        vector_results = self._vector_search(query, top_k, namespace, filter)
        logger.info(f"  Vector search: {len(vector_results)} results")
        
        # Step 2: Extract entities from query and retrieved passages
        entities = self._extract_entities(query, vector_results)
        logger.info(f"  Entities found: {entities}")
        
        # Step 3: Graph traversal for each entity
        graph_results = []
        for entity in entities[:5]:  # Cap at 5 entities
            subgraph = self.kg.query_subgraph(entity, max_depth=graph_depth)
            if subgraph.nodes:  # Only include non-empty results
                graph_results.append(subgraph)
        logger.info(f"  Graph traversals: {len(graph_results)} subgraphs")
        
        # Step 4: Merge results
        merged_context = self._merge_contexts(query, vector_results, graph_results)
        
        # Step 5: Generate answer
        answer = self._generate_answer(query, merged_context)
        
        return HybridResult(
            query=query,
            vector_results=[{
                "id": r.id,
                "score": round(r.score, 3),
                "excerpt": r.content[:300],
                "metadata": r.metadata or {},
            } for r in vector_results],
            graph_results=graph_results,
            merged_context=merged_context,
            answer=answer,
            retrieval_metadata={
                "vector_count": len(vector_results),
                "graph_subgraphs": len(graph_results),
                "entities_queried": entities,
                "vector_weight": self.vector_weight,
                "graph_weight": self.graph_weight,
            },
        )
    
    def _vector_search(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filter: Optional[Dict[str, Any]],
    ) -> List[VectorSearchResult]:
        """Perform vector similarity search via configured store."""
        query_result = self.embeddings.embed_text(query)

        if not query_result.embedding:
            return []

        return self.vector_store.search(
            query_vector=query_result.embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        )

    def _extract_entities(
        self,
        query: str,
        vector_results: List[VectorSearchResult],
    ) -> List[str]:
        """Extract entity names from query and retrieved passages."""
        # Extract from query
        entities = self.kg.extract_entities_from_query(query)
        
        # Also extract from top retrieved passages
        if vector_results:
            combined_text = " ".join([r.content[:200] for r in vector_results[:3]])
            passage_entities = self.kg.extract_entities_from_query(combined_text)
            
            # Merge, preserving order and removing duplicates
            seen = set(e.lower() for e in entities)
            for e in passage_entities:
                if e.lower() not in seen:
                    entities.append(e)
                    seen.add(e.lower())
        
        return entities
    
    def _merge_contexts(
        self,
        query: str,
        vector_results: List[VectorSearchResult],
        graph_results: List[SubgraphResult],
    ) -> str:
        """
        Merge vector search results with knowledge graph context.
        
        Produces a unified context string with clear section markers
        so the LLM knows which information comes from which source.
        """
        sections = []
        
        # Vector search context
        if vector_results:
            sections.append("=== RETRIEVED DOCUMENT PASSAGES (Semantic Search) ===")
            for i, r in enumerate(vector_results):
                case_name = (r.metadata or {}).get("case_name", "Unknown")
                sections.append(f"\n[Passage {i+1}, Score: {r.score:.3f}, Case: {case_name}]")
                sections.append(r.content)
        
        # Knowledge graph context
        if graph_results:
            sections.append("\n\n=== KNOWLEDGE GRAPH CONTEXT (Structured Relationships) ===")
            for subgraph in graph_results:
                sections.append(f"\n{subgraph.linearized_text}")
        
        return "\n".join(sections)
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using the merged context."""
        prompt = f"""You are a legal research assistant with access to both document passages
and a structured knowledge graph of legal relationships.

QUESTION: {query}

CONTEXT:
{context}

Using BOTH the document passages AND the knowledge graph relationships, provide a
comprehensive legal analysis. When the knowledge graph shows case citation chains or
factor relationships, explicitly trace those connections in your answer.

Structure your response with:
1. Direct answer based on document passages
2. Relationship insights from the knowledge graph (precedent chains, factor linkages)
3. Synthesis combining both sources
4. Confidence assessment and any remaining gaps"""

        system_instruction = """You are a legal research assistant for the Deel Lab for Global Employment.
You have access to both semantic search results and structured knowledge graph data.
When graph relationships add context that document passages don't explicitly state,
highlight this as structural insight from the knowledge graph."""

        try:
            return self.chat.generate(
                prompt,
                system_instruction=system_instruction,
                temperature=0.4,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return f"Error generating answer: {str(e)}"


def main():
    """Test the hybrid retriever"""
    print("\n" + "=" * 60)
    print("TESTING HYBRID RETRIEVER")
    print("=" * 60)
    
    try:
        retriever = HybridRetriever()
        
        # Try to load existing knowledge graph
        retriever.kg.load()
        
        result = retriever.retrieve(
            "How does the Sagaz test apply to worker classification in Ontario?",
            top_k=3,
        )
        
        print(f"\n📝 Query: {result.query}")
        print(f"\n📊 Vector results: {len(result.vector_results)}")
        print(f"🔗 Graph subgraphs: {len(result.graph_results)}")
        print(f"\n💬 Answer:\n{result.answer[:500]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
