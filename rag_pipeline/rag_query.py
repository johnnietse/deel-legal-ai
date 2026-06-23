# RAG Pipeline - Query Interface
"""
RAG query interface for legal document retrieval and response generation.

Combines:
- Semantic search via Pinecone
- Context assembly with source citations
- Response generation via Gemini
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import GeminiEmbeddings, GeminiChat
from rag_pipeline.pinecone_client import PineconeClient, SearchResult
from rag_pipeline.verifier import ResponseVerifier

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG query"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    verification: Optional[Dict[str, Any]] = None  # Attached when verify=True
    

class LegalRAGQuery:
    """
    RAG query interface for legal document Q&A.
    
    Features:
    - Semantic search over legal documents
    - Context-aware response generation
    - Source citation formatting
    - Legal section filtering
    """
    
    def __init__(
        self,
        embeddings: Optional[GeminiEmbeddings] = None,
        chat: Optional[GeminiChat] = None,
        pinecone: Optional[PineconeClient] = None
    ):
        self.embeddings = embeddings or GeminiEmbeddings()
        self.chat = chat or GeminiChat()
        self.pinecone = pinecone or PineconeClient()
        self.verifier = ResponseVerifier(chat=self.chat)
        
        # Ensure Pinecone is connected
        self.pinecone.connect()
    
    def _format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format search results as source citations"""
        sources = []
        for i, result in enumerate(results):
            source = {
                "index": i + 1,
                "id": result.id,
                "score": round(result.score, 3),
                "excerpt": result.content[:500] + "..." if len(result.content) > 500 else result.content,
            }
            
            # Add metadata
            if result.metadata:
                source["case_name"] = result.metadata.get("case_name", "Unknown")
                source["citation"] = result.metadata.get("primary_citation", "")
                source["court"] = result.metadata.get("court", "")
                source["jurisdiction"] = result.metadata.get("jurisdiction", "")
                source["legal_section"] = result.metadata.get("legal_section", "")
            
            sources.append(source)
        
        return sources
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_analysis: bool = True,
        verify: bool = False
    ) -> RAGResponse:
        """
        Query the legal knowledge base.
        
        Args:
            question: Legal question to answer
            top_k: Number of relevant documents to retrieve
            namespace: Pinecone namespace to search
            filter: Metadata filter (e.g., {"jurisdiction": "ON"})
            include_analysis: Whether to generate analysis response
            
        Returns:
            RAGResponse with answer and sources
        """
        # Generate query embedding
        logger.info(f"Processing query: {question[:50]}...")
        query_result = self.embeddings.embed_text(question)
        
        if not query_result.embedding:
            logger.error("Failed to generate query embedding")
            return RAGResponse(
                query=question,
                answer="Error: Could not process your question. Please try again.",
                sources=[],
                confidence="low"
            )
        
        # Search Pinecone
        search_results = self.pinecone.search(
            query_vector=query_result.embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter
        )
        
        if not search_results:
            return RAGResponse(
                query=question,
                answer="No relevant legal documents found in the knowledge base for your question.",
                sources=[],
                confidence="low"
            )
        
        # Format sources
        sources = self._format_sources(search_results)
        
        # Determine confidence based on top scores
        avg_score = sum(r.score for r in search_results) / len(search_results)
        confidence = "high" if avg_score > 0.8 else "medium" if avg_score > 0.6 else "low"
        
        # Generate response if requested
        if include_analysis:
            context = [r.content for r in search_results]
            answer = self.chat.generate_with_context(question, context)
            
            verification_report = None
            if verify:
                logger.info("Running post-hoc verification on generated answer...")
                verification = self.verifier.verify_grounding(answer, sources)
                verification_report = verification.to_dict()
                if not verification.is_grounded:
                    logger.warning(f"Verification failed! Unsupported claims: {verification.unsupported_claims}")
                    if verification.corrected_answer:
                        answer = f"{verification.corrected_answer}\n\n[System Note: The original response was modified because it contained claims unsupported by the retrieved documents.]"
                    else:
                        answer = f"{answer}\n\n[System Warning: The verification module flagged the following claims as potentially unsupported by the documents: {', '.join(verification.unsupported_claims)}]"
        else:
            answer = "See sources below for relevant legal information."
            verification_report = None
        
        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            verification=verification_report,
        )
    
    def query_multi_hop(
        self,
        question: str,
        max_hops: int = 5,
        top_k_per_hop: int = 3,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        verify: bool = False,
    ) -> RAGResponse:
        """
        Query using multi-hop retrieval for complex questions.
        
        Uses iterative retrieve-read-reason cycles to build a complete
        evidence chain before generating the final answer.
        
        Args:
            question: Complex legal question
            max_hops: Maximum retrieval hops
            top_k_per_hop: Documents per hop
            namespace: Pinecone namespace
            filter: Metadata filter
            
        Returns:
            RAGResponse with multi-hop answer and sources
        """
        from rag_pipeline.multi_hop_retriever import MultiHopRetriever
        
        retriever = MultiHopRetriever(
            embeddings=self.embeddings,
            chat=self.chat,
            pinecone=self.pinecone,
            max_hops=max_hops,
        )
        
        result = retriever.retrieve(
            question,
            top_k_per_hop=top_k_per_hop,
            namespace=namespace,
            filter=filter,
        )
        
        # Convert to standard RAGResponse format
        sources = []
        for i, source in enumerate(result.sources):
            sources.append({
                "index": i + 1,
                "id": source.get("id", ""),
                "score": source.get("score", 0.0),
                "excerpt": source.get("content", "")[:500],
                "hop": source.get("hop", 1),
                "case_name": source.get("metadata", {}).get("case_name", "Unknown"),
            })
        
        confidence = "high" if result.final_completeness > 0.8 else \
                     "medium" if result.final_completeness > 0.5 else "low"
        
        answer = result.answer
        verification_report = None
        if verify:
            logger.info("Running post-hoc verification on multi-hop answer...")
            verification = self.verifier.verify_grounding(answer, sources)
            verification_report = verification.to_dict()
            if not verification.is_grounded:
                if verification.corrected_answer:
                    answer = f"{verification.corrected_answer}\n\n[System Note: The original response was modified because it contained claims unsupported by the retrieved documents.]"
                else:
                    answer = f"{answer}\n\n[System Warning: The verification module flagged the following claims as potentially unsupported by the documents: {', '.join(verification.unsupported_claims)}]"
        
        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            verification=verification_report,
        )
    
    def query_smart(
        self,
        question: str,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        verify: bool = False,
    ) -> RAGResponse:
        """
        Smart query that auto-routes between single-hop and multi-hop
        based on question complexity.
        
        Uses a simple heuristic: questions with multiple clauses,
        cross-references, or comparative elements get multi-hop.
        """
        complexity = self._estimate_complexity(question)
        
        if complexity == "complex":
            logger.info("Auto-routing to multi-hop retrieval")
            return self.query_multi_hop(question, namespace=namespace, filter=filter, verify=verify)
        else:
            logger.info("Auto-routing to single-hop retrieval")
            return self.query(question, namespace=namespace, filter=filter, verify=verify)
    
    def _estimate_complexity(self, question: str) -> str:
        """
        Estimate question complexity for routing decisions.
        
        Returns "simple" or "complex" based on heuristics.
        """
        complexity_indicators = [
            "and also", "in addition", "compared to", "versus",
            "how does", "what are all", "comprehensive",
            "multiple", "different", "interact", "relationship between",
            "both", "as well as", "furthermore", "across",
        ]
        
        question_lower = question.lower()
        indicator_count = sum(1 for ind in complexity_indicators if ind in question_lower)
        word_count = len(question.split())
        
        # Complex if: many words, multiple indicators, or question marks
        if indicator_count >= 2 or word_count > 40 or question.count("?") > 1:
            return "complex"
        
        return "simple"
    
    def query_worker_classification(
        self,
        facts: str,
        jurisdiction: str = "ON"
    ) -> RAGResponse:
        """
        Specialized query for worker classification analysis.
        
        Args:
            facts: Description of the working relationship
            jurisdiction: Legal jurisdiction (default: Ontario)
            
        Returns:
            RAGResponse with classification analysis
        """
        # Build specialized prompt
        prompt = f"""Analyze the following worker classification scenario under {jurisdiction} employment law:

FACTS:
{facts}

Based on the legal precedents in the knowledge base, analyze:
1. Whether the worker is likely an employee or independent contractor
2. Key factors supporting this classification
3. Relevant legal tests (e.g., Sagaz test)
4. Potential risks if misclassified"""
        
        return self.query(
            question=prompt,
            top_k=8,
            filter={"jurisdiction": jurisdiction} if jurisdiction else None
        )


def main():
    """Test the RAG query interface"""
    print("\n" + "="*60)
    print("TESTING LEGAL RAG QUERY")
    print("="*60)
    
    try:
        rag = LegalRAGQuery()
        
        # Test query
        response = rag.query(
            "What factors determine if a worker is an employee or independent contractor in Ontario?",
            top_k=3
        )
        
        print(f"\n📝 Query: {response.query}")
        print(f"\n📊 Confidence: {response.confidence}")
        print(f"\n📖 Sources: {len(response.sources)}")
        
        for source in response.sources:
            print(f"\n   Source {source['index']}: {source.get('case_name', 'Unknown')}")
            print(f"   Score: {source['score']}")
            print(f"   Excerpt: {source['excerpt'][:100]}...")
        
        print(f"\n💬 Answer:\n{response.answer}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
