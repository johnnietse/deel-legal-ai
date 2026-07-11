# RAG Pipeline - Query Interface (ByteDance Enhanced)
"""
RAG query interface for legal document retrieval and response generation.

Enhanced with ByteDance RAG best practices:
  - Hybrid retrieval (BM25 + Vector via Elasticsearch + Pinecone/Milvus)
  - Structured prompt templates with auto-selection
  - Pre-generation confidence gate
  - Multi-layer caching (embedding, retrieval, response)
  - Full-pipeline metrics instrumentation
  - Post-hoc verification (existing)
  - Multi-hop retrieval (existing)
  - Smart routing (existing)
"""

import sys
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import GeminiEmbeddings, GeminiChat
from rag_pipeline.vector_store import create_vector_store, VectorSearchResult
from rag_pipeline.verifier import ResponseVerifier
from rag_pipeline.hybrid_retriever import HybridRetriever, HybridResult
from rag_pipeline.prompt_templates import PromptTemplateLibrary
from rag_pipeline.confidence_gate import ConfidenceGate
from rag_pipeline.query_cache import RAGQueryCache
from rag_pipeline.metrics import QueryMetrics, MetricsCollector, timed_stage

import config
from config import (
    MULTI_GRANULARITY_SEARCH_ENABLED, MULTI_GRANULARITY_ENABLED,
    DOCUMENT_SUMMARY_NAMESPACE, CHUNK_NAMESPACE,
    HYBRID_DEFAULT_TOP_K,
)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG query"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: str
    verification: Optional[Dict[str, Any]] = None
    confidence_gate: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    retrieval_mode: str = "hybrid"
    template_used: str = ""
    

class LegalRAGQuery:
    """
    RAG query interface for legal document Q&A.
    
    ByteDance-enhanced features:
    - Hybrid BM25 + vector retrieval with query-type-aware weights
    - Structured prompt templates (worker classification, notice period, etc.)
    - Pre-generation confidence gate (refuse/hedge low-confidence answers)
    - Multi-layer caching (embedding, retrieval, response)
    - Full-pipeline metrics (latency, cost, quality)
    - Post-hoc verification (NLI-based fact-checking)
    - Multi-hop retrieval for complex questions
    - Smart routing (auto single-hop vs multi-hop)
    """
    
    def __init__(
        self,
        embeddings: Optional[GeminiEmbeddings] = None,
        chat: Optional[GeminiChat] = None,
        vector_store=None,
        enable_cache: bool = True,
        enable_confidence_gate: bool = True,
        enable_metrics: bool = True,
    ):
        # Core services
        self.embeddings = embeddings or GeminiEmbeddings()
        self.chat = chat or GeminiChat()
        self.vector_store = vector_store if vector_store is not None else create_vector_store()
        self.verifier = ResponseVerifier(chat=self.chat)
        
        # Ensure vector store is connected
        self.vector_store.connect()
        
        # ByteDance enhancements
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            fusion_method=config.HYBRID_FUSION_METHOD,
            mmr_lambda=config.HYBRID_MMR_LAMBDA,
            default_top_k=config.HYBRID_DEFAULT_TOP_K,
        )
        
        # Build BM25 index from vector store for hybrid search
        try:
            self.hybrid_retriever.build_bm25_from_vector_store(namespace=CHUNK_NAMESPACE)
        except Exception as e:
            logger.warning(f"Failed to build BM25 index from vector store: {e}")
        
        self.prompt_library = PromptTemplateLibrary()
        
        self.confidence_gate = ConfidenceGate(
            refuse_threshold=config.CONFIDENCE_REFUSE_THRESHOLD,
            hedge_threshold=config.CONFIDENCE_HEDGE_THRESHOLD,
        ) if enable_confidence_gate and config.CONFIDENCE_GATE_ENABLED else None
        
        self.cache = RAGQueryCache(
            cache_dir=config.CACHE_DIR,
            embedding_ttl=config.CACHE_EMBEDDING_TTL,
            retrieval_ttl=config.CACHE_RETRIEVAL_TTL,
            response_ttl=config.CACHE_RESPONSE_TTL,
        ) if enable_cache and config.CACHE_ENABLED else None
        
        self.metrics_collector = MetricsCollector(
            log_dir=config.METRICS_LOG_DIR,
        ) if enable_metrics else None
    
    def _format_sources(self, results) -> List[Dict[str, Any]]:
        """Format search results as source citations (supports both types)."""
        sources = []
        for i, result in enumerate(results):
            # Handle both SearchResult and HybridResult
            if isinstance(result, HybridResult):
                source = {
                    "index": i + 1,
                    "id": result.id,
                    "score": round(result.score, 3),
                    "excerpt": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    "content": result.content,
                    "bm25_score": round(result.bm25_score, 3),
                    "vector_score": round(result.vector_score, 3),
                    "retrieved_by": result.retrieved_by,
                }
                metadata = result.metadata
            else:
                source = {
                    "index": i + 1,
                    "id": result.id,
                    "score": round(result.score, 3),
                    "excerpt": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    "content": result.content,
                }
                metadata = result.metadata
            
            # Add metadata
            if metadata:
                source["case_name"] = metadata.get("title", metadata.get("case_name", "Unknown"))
                source["citation"] = metadata.get("citation", metadata.get("primary_citation", ""))
                source["court"] = metadata.get("court", "")
                source["jurisdiction"] = metadata.get("jurisdiction", "")
                source["legal_section"] = metadata.get("legal_section", "")
            
            sources.append(source)
        
        return sources
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_analysis: bool = True,
        verify: bool = False,
        template_name: Optional[str] = None,
        force_retrieval_mode: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> RAGResponse:
        """
        Query the legal knowledge base with full ByteDance-enhanced pipeline.
        
        Pipeline stages:
          1. [Cache] Check response cache for exact match
          2. [Retrieval] Hybrid BM25 + vector search (query-type-aware)
          3. [Template] Auto-select prompt template by query intent
          4. [Generation] Generate answer using structured prompt
          5. [Confidence] Post-gen confidence gate (refuse/hedge/pass)
          6. [Verification] Optional NLI-based fact-checking
          7. [Metrics] Log structured metrics
          8. [Cache] Store result in cache
        
        Args:
            question: Legal question to answer
            top_k: Number of relevant documents to retrieve
            namespace: Pinecone namespace to search
            filter: Metadata filter (e.g., {"jurisdiction": "ON"})
            include_analysis: Whether to generate analysis response
            verify: Run post-hoc verification
            template_name: Force a specific prompt template
            force_retrieval_mode: Override auto retrieval ("bm25", "vector", "hybrid")
            
        Returns:
            RAGResponse with answer, sources, confidence, and metrics
        """
        query_id = str(uuid.uuid4())[:8]
        metrics = QueryMetrics(query_id=query_id, query_text=question[:200], user_id=user_id or "")
        total_start = time.perf_counter()
        
        # --- Stage 1: Response Cache Check ---
        if self.cache:
            cached = self.cache.get_response(question, top_k=top_k, namespace=namespace)
            if cached:
                logger.info(f"[{query_id}] Cache hit for response")
                metrics.generation_cache_hit = True
                metrics.total_latency_ms = (time.perf_counter() - total_start) * 1000
                if self.metrics_collector:
                    self.metrics_collector.record(metrics)
                cached_resp = RAGResponse(**cached)
                cached_resp.metrics = metrics.to_dict()
                return cached_resp
        
        # --- Stage 2: Hybrid Retrieval ---
        with timed_stage("retrieval", metrics):
            if config.HYBRID_SEARCH_ENABLED:
                search_results = self.hybrid_retriever.retrieve(
                    query=question,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter,
                    force_mode=force_retrieval_mode,
                )
                metrics.retrieval_mode = "hybrid"
            else:
                # Fallback to pure vector search
                query_result = self.embeddings.embed_text(question)
                if not query_result.embedding:
                    return RAGResponse(
                        query=question,
                        answer="Error: Could not process your question.",
                        sources=[], confidence="low",
                    )
                raw_results = self.vector_store.search(
                    query_vector=query_result.embedding,
                    top_k=top_k, namespace=namespace, filter=filter,
                )
                search_results = raw_results
                metrics.retrieval_mode = "vector"
        
        metrics.retrieval_result_count = len(search_results)
        if search_results:
            scores = [r.score for r in search_results]
            metrics.retrieval_avg_score = sum(scores) / len(scores)
        
        # --- Multi-Granularity Search: Supplement with doc-level results (ByteDance §4.1.2) ---
        if MULTI_GRANULARITY_SEARCH_ENABLED and MULTI_GRANULARITY_ENABLED:
            try:
                doc_ns = DOCUMENT_SUMMARY_NAMESPACE
                query_embed = self.embeddings.embed_text(question)
                if query_embed.embedding:
                    doc_level_results = self.vector_store.search(
                        query_vector=query_embed.embedding,
                        top_k=top_k,
                        namespace=doc_ns,
                        filter=filter,
                    )
                    if doc_level_results:
                        # Tag doc-level results so they're distinguishable
                        for r in doc_level_results:
                            if not hasattr(r, 'retrieved_by'):
                                r.retrieved_by = ["vector_doc_level"]
                            else:
                                r.retrieved_by.append("vector_doc_level")
                        # Merge: dedup by ID, interleave doc-level results (which provide broader context)
                        existing_ids = {r.id for r in search_results}
                        doc_results_filtered = [r for r in doc_level_results if r.id not in existing_ids]
                        if doc_results_filtered:
                            # Insert doc-level results at the bottom of the main list
                            search_results = list(search_results) + doc_results_filtered
                            # Re-sort by score descending
                            search_results.sort(key=lambda r: r.score, reverse=True)
                            logger.info(
                                f"Multi-granularity: added {len(doc_results_filtered)} "
                                f"doc-level results from namespace '{doc_ns}'"
                            )
                            metrics.retrieval_result_count = len(search_results)
            except Exception as e:
                logger.warning(f"Multi-granularity search failed (non-fatal): {e}")
        
        if not search_results:
            return RAGResponse(
                query=question,
                answer="No relevant legal documents found in the knowledge base.",
                sources=[], confidence="low",
            )
        
        # Format sources
        sources = self._format_sources(search_results)
        
        # Determine confidence based on top scores
        avg_score = metrics.retrieval_avg_score
        confidence = "high" if avg_score > 0.8 else "medium" if avg_score > 0.6 else "low"
        
        # --- Stage 3 & 4: Template Selection + Generation ---
        if include_analysis:
            with timed_stage("generation", metrics):
                # Auto-select prompt template
                system_instruction, user_prompt = self.prompt_library.build_prompt(
                    query=question,
                    sources=sources,
                    template_name=template_name,
                    max_sources=config.PROMPT_MAX_SOURCES,
                )
                
                template_used = template_name or self.prompt_library.auto_select(question).name
                metrics.generation_template_used = template_used
                
                # Estimate input tokens
                metrics.generation_input_tokens = len(
                    (system_instruction + user_prompt).split()
                )
                metrics.generation_model = self.chat.model
                
                # Generate answer
                answer = self.chat.generate(
                    user_prompt,
                    system_instruction=system_instruction,
                )
                
                metrics.generation_output_tokens = len(answer.split())
            
            # --- Stage 5: Confidence Gate ---
            confidence_report = None
            if self.confidence_gate:
                gate_result = self.confidence_gate.check(answer, sources, question)
                confidence_report = gate_result.to_dict()
                metrics.confidence_score = gate_result.confidence_score
                metrics.confidence_action = gate_result.action_taken
                
                if not gate_result.passed and gate_result.modified_answer:
                    answer = gate_result.modified_answer
                    logger.info(
                        f"[{query_id}] Confidence gate: {gate_result.action_taken} "
                        f"(score={gate_result.confidence_score:.2f})"
                    )
            
            # --- Stage 6: Verification (optional) ---
            verification_report = None
            if verify:
                with timed_stage("verification", metrics):
                    logger.info(f"[{query_id}] Running post-hoc verification...")
                    verification = self.verifier.verify_grounding(answer, sources)
                    verification_report = verification.to_dict()
                    
                    metrics.verification_grounding_score = verification.grounding_score
                    metrics.verification_claim_count = len(verification.claims)
                    metrics.verification_hallucination_count = len(verification.unsupported_claims)
                    
                    if not verification.is_grounded:
                        logger.warning(
                            f"[{query_id}] Verification failed! "
                            f"Unsupported: {verification.unsupported_claims}"
                        )
                        if verification.corrected_answer:
                            answer = (
                                f"{verification.corrected_answer}\n\n"
                                "[System Note: The original response was modified because "
                                "it contained claims unsupported by the retrieved documents.]"
                            )
                        else:
                            answer = (
                                f"{answer}\n\n"
                                "[System Warning: The verification module flagged the "
                                f"following claims as potentially unsupported: "
                                f"{', '.join(verification.unsupported_claims)}]"
                            )
        else:
            answer = "See sources below for relevant legal information."
            verification_report = None
            confidence_report = None
            template_used = ""
        
        # --- Stage 7: Metrics ---
        metrics.total_latency_ms = (time.perf_counter() - total_start) * 1000
        if self.metrics_collector:
            self.metrics_collector.record(metrics)
        
        response = RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            verification=verification_report,
            confidence_gate=confidence_report,
            metrics=metrics.to_dict(),
            retrieval_mode=metrics.retrieval_mode,
            template_used=template_used if include_analysis else "",
        )
        
        # --- Stage 8: Cache Store ---
        if self.cache and include_analysis:
            self.cache.put_response(
                question,
                {
                    "query": response.query,
                    "answer": response.answer,
                    "sources": response.sources,
                    "confidence": response.confidence,
                    "retrieval_mode": response.retrieval_mode,
"template_used": response.template_used,
                },
                top_k=top_k, namespace=namespace,
            )
        
        return response

    def query_multi_hop(
        self,
        question: str,
        max_hops: int = 5,
        top_k_per_hop: int = 3,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        verify: bool = False,
        user_id: Optional[str] = None,
    ) -> RAGResponse:
        """
        Query using multi-hop retrieval for complex questions.
        
        Uses iterative retrieve-read-reason cycles to build a complete
        evidence chain before generating the final answer.
        """
        from rag_pipeline.multi_hop_retriever import MultiHopRetriever
        import time
        import uuid
        from rag_pipeline.metrics import QueryMetrics
        
        # Initialize metrics
        query_id = str(uuid.uuid4())[:8]
        metrics = QueryMetrics(query_id=query_id, query_text=question[:200], user_id=user_id or "")
        total_start = time.perf_counter()
        
        retriever = MultiHopRetriever(
            embeddings=self.embeddings,
            chat=self.chat,
            vector_store=self.vector_store,
            hybrid_retriever=self.hybrid_retriever if config.HYBRID_SEARCH_ENABLED else None,
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
        
        # Record metrics
        metrics.total_latency_ms = (time.perf_counter() - total_start) * 1000
        metrics.retrieval_mode = "multi_hop"
        if self.metrics_collector:
            self.metrics_collector.record(metrics)
        
        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            verification=verification_report,
            retrieval_mode="multi_hop",
metrics=metrics.to_dict(),
        )
    
    def query_smart(
        self,
        question: str,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        verify: bool = False,
        user_id: Optional[str] = None,
    ) -> RAGResponse:
        """
        Smart query that auto-routes between single-hop and multi-hop
        based on question complexity.
        """
        complexity = self._estimate_complexity(question)
        
        if complexity == "complex":
            logger.info("Auto-routing to multi-hop retrieval")
            return self.query_multi_hop(question, namespace=namespace, filter=filter, verify=verify, user_id=user_id)
        else:
            logger.info("Auto-routing to single-hop retrieval")
            return self.query(question, namespace=namespace, filter=filter, verify=verify, user_id=user_id)
    
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
        
        Now uses the worker_classification prompt template automatically.
        """
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
            filter={"jurisdiction": jurisdiction} if jurisdiction else None,
            template_name="worker_classification",
        )
    
    # -- Pipeline stats ---------------------------------------------------
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get full pipeline statistics (ByteDance §8.1)."""
        stats = {}
        
        if self.metrics_collector:
            stats["metrics"] = self.metrics_collector.summary()
        
        if self.cache:
            stats["cache"] = self.cache.stats()
        
        return stats


def main():
    """Test the RAG query interface"""
    print("\n" + "="*60)
    print("TESTING LEGAL RAG QUERY (ByteDance Enhanced)")
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
        print(f"🔍 Retrieval Mode: {response.retrieval_mode}")
        print(f"📋 Template Used: {response.template_used}")
        print(f"\n📖 Sources: {len(response.sources)}")
        
        for source in response.sources:
            print(f"\n   Source {source['index']}: {source.get('case_name', 'Unknown')}")
            print(f"   Score: {source['score']}")
            print(f"   Excerpt: {source['excerpt'][:100]}...")
        
        print(f"\n💬 Answer:\n{response.answer}")
        
        if response.confidence_gate:
            print(f"\n🛡️ Confidence Gate: {response.confidence_gate}")
        
        if response.metrics:
            print(f"\n📊 Metrics: {response.metrics}")
        
        # Pipeline stats
        stats = rag.get_pipeline_stats()
        print(f"\n📈 Pipeline Stats: {stats}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
