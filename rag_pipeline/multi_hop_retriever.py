# RAG Pipeline - Multi-Hop Retriever
"""
Multi-hop reasoning retrieval for complex legal questions.

Instead of a single vector search → answer, this module implements an iterative
retrieve-read-reason cycle that:

1. Retrieves initial documents
2. Reads them and identifies information gaps
3. Generates new sub-queries to fill those gaps
4. Repeats until evidence chain is complete
5. Generates a final answer with the full evidence chain

This directly addresses the research challenge described as "检索轨迹规划"
(retrieval trajectory planning) — designing the optimal state machine flow
for multi-hop retrieval in complex legal analysis.

Key research contributions:
- Gap detection function: identifying missing information after each hop
- Query reformulation: generating targeted sub-queries from identified gaps
- Stopping criterion: when to stop retrieving vs. continue searching
- Evidence chain assembly: combining multi-hop results into coherent context
"""

import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import GeminiEmbeddings, GeminiChat
from rag_pipeline.pinecone_client import PineconeClient, SearchResult
from config import (
    MULTI_HOP_MAX_HOPS,
    MULTI_HOP_COMPLETENESS_THRESHOLD,
    MULTI_HOP_MIN_NEW_INFO_TOKENS,
)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RetrievalHop:
    """Record of a single retrieval hop in the multi-hop chain"""
    hop_number: int
    query: str
    retrieved_docs: List[Dict[str, Any]]
    identified_gaps: List[str]
    completeness_score: float
    new_information: str
    reasoning: str


@dataclass
class RetrievalState:
    """
    State machine node for tracking multi-hop retrieval progress.
    
    This is the core state representation for the retrieval trajectory.
    The state machine transitions are:
        INIT -> RETRIEVE -> ANALYZE -> (CONTINUE | COMPLETE)
        CONTINUE -> RETRIEVE -> ANALYZE -> (CONTINUE | COMPLETE)
    """
    original_query: str
    current_sub_query: str
    hops: List[RetrievalHop] = field(default_factory=list)
    accumulated_evidence: List[str] = field(default_factory=list)
    accumulated_sources: List[Dict[str, Any]] = field(default_factory=list)
    remaining_gaps: List[str] = field(default_factory=list)
    overall_completeness: float = 0.0
    status: str = "init"  # init, retrieving, analyzing, complete, max_hops_reached


@dataclass
class MultiHopResult:
    """Final result of a multi-hop retrieval"""
    query: str
    answer: str
    evidence_chain: List[RetrievalHop]
    total_hops: int
    final_completeness: float
    sources: List[Dict[str, Any]]
    reasoning_trace: str
    duration_ms: float


class MultiHopRetriever:
    """
    Multi-hop retrieval engine with gap detection and query reformulation.
    
    Implements the "检索轨迹规划" (retrieval trajectory planning) pattern:
    - Each hop is a state transition in a retrieval state machine
    - Gap detection identifies what information is still missing
    - Query reformulation generates targeted sub-queries for each gap
    - Stopping criterion balances completeness against diminishing returns
    
    Research knobs to tune:
    - max_hops: Maximum number of retrieval hops (depth budget)
    - completeness_threshold: When evidence is "good enough" to stop
    - min_new_info_tokens: Minimum new information per hop to continue
    """
    
    def __init__(
        self,
        embeddings: Optional[GeminiEmbeddings] = None,
        chat: Optional[GeminiChat] = None,
        pinecone: Optional[PineconeClient] = None,
        max_hops: int = MULTI_HOP_MAX_HOPS,
        completeness_threshold: float = MULTI_HOP_COMPLETENESS_THRESHOLD,
        min_new_info_tokens: int = MULTI_HOP_MIN_NEW_INFO_TOKENS,
    ):
        self.embeddings = embeddings or GeminiEmbeddings()
        self.chat = chat or GeminiChat()
        self.pinecone = pinecone or PineconeClient()
        self.max_hops = max_hops
        self.completeness_threshold = completeness_threshold
        self.min_new_info_tokens = min_new_info_tokens
        
        # Ensure Pinecone is connected
        self.pinecone.connect()
    
    def retrieve(
        self,
        question: str,
        top_k_per_hop: int = 3,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
    ) -> MultiHopResult:
        """
        Perform multi-hop retrieval for a complex legal question.
        
        The retrieval loop:
        1. Generate embedding for current query
        2. Search Pinecone for relevant documents
        3. Read retrieved documents and analyze gaps
        4. If gaps remain and hops < max, reformulate query and go to 1
        5. Otherwise, assemble full evidence chain and generate answer
        
        Args:
            question: Complex legal question requiring multi-hop reasoning
            top_k_per_hop: Number of documents to retrieve per hop
            namespace: Pinecone namespace
            filter: Optional metadata filter
            
        Returns:
            MultiHopResult with full evidence chain and reasoning trace
        """
        start_time = datetime.now()
        
        state = RetrievalState(
            original_query=question,
            current_sub_query=question,
        )
        
        logger.info(f"Starting multi-hop retrieval for: {question[:80]}...")
        
        for hop_num in range(1, self.max_hops + 1):
            state.status = "retrieving"
            logger.info(f"  Hop {hop_num}: querying '{state.current_sub_query[:60]}...'")
            
            # Step 1: Retrieve documents for current sub-query
            retrieved = self._retrieve_documents(
                state.current_sub_query, top_k_per_hop, namespace, filter
            )
            
            if not retrieved:
                logger.warning(f"  Hop {hop_num}: No documents retrieved")
                break
            
            # Step 2: Analyze retrieved documents for gaps
            state.status = "analyzing"
            analysis = self._analyze_and_detect_gaps(
                state.original_query,
                state.current_sub_query,
                retrieved,
                state.accumulated_evidence,
            )
            
            # Step 3: Record this hop
            hop = RetrievalHop(
                hop_number=hop_num,
                query=state.current_sub_query,
                retrieved_docs=[{
                    "id": r.id,
                    "score": round(r.score, 3),
                    "excerpt": r.content[:300],
                    "metadata": r.metadata or {},
                } for r in retrieved],
                identified_gaps=analysis["gaps"],
                completeness_score=analysis["completeness"],
                new_information=analysis["new_info"],
                reasoning=analysis["reasoning"],
            )
            state.hops.append(hop)
            
            # Step 4: Accumulate evidence
            for r in retrieved:
                if r.content not in state.accumulated_evidence:
                    state.accumulated_evidence.append(r.content)
                    state.accumulated_sources.append({
                        "id": r.id,
                        "score": round(r.score, 3),
                        "content": r.content,
                        "metadata": r.metadata or {},
                        "hop": hop_num,
                    })
            
            state.remaining_gaps = analysis["gaps"]
            state.overall_completeness = analysis["completeness"]
            
            # Step 5: Decide whether to continue
            should_stop, stop_reason = self._should_stop(state, analysis)
            
            if should_stop:
                logger.info(f"  Stopping at hop {hop_num}: {stop_reason}")
                state.status = "complete"
                break
            
            # Step 6: Generate next sub-query from the most critical gap
            if analysis["gaps"]:
                state.current_sub_query = self._generate_sub_query(
                    analysis["gaps"][0],
                    state.original_query,
                    state.accumulated_evidence,
                )
                logger.info(f"  Next sub-query: {state.current_sub_query[:60]}...")
        else:
            state.status = "max_hops_reached"
            logger.info(f"  Max hops ({self.max_hops}) reached")
        
        # Step 7: Generate final answer with full evidence chain
        answer = self._generate_final_answer(
            state.original_query,
            state.accumulated_evidence,
            state.hops,
        )
        
        # Build reasoning trace
        reasoning_trace = self._build_reasoning_trace(state)
        
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return MultiHopResult(
            query=question,
            answer=answer,
            evidence_chain=state.hops,
            total_hops=len(state.hops),
            final_completeness=state.overall_completeness,
            sources=state.accumulated_sources,
            reasoning_trace=reasoning_trace,
            duration_ms=round(duration_ms, 1),
        )
    
    def _retrieve_documents(
        self,
        query: str,
        top_k: int,
        namespace: str,
        filter: Optional[Dict[str, Any]],
    ) -> List[SearchResult]:
        """Retrieve documents from Pinecone for a given query."""
        query_result = self.embeddings.embed_text(query)
        
        if not query_result.embedding:
            return []
        
        return self.pinecone.search(
            query_vector=query_result.embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
        )
    
    def _analyze_and_detect_gaps(
        self,
        original_query: str,
        current_query: str,
        retrieved: List[SearchResult],
        prior_evidence: List[str],
    ) -> Dict[str, Any]:
        """
        Use Gemini to analyze retrieved documents and identify information gaps.
        
        This is a key research function — the gap detection quality directly
        determines the effectiveness of multi-hop retrieval.
        
        Returns:
            Dict with keys: gaps, completeness, new_info, reasoning
        """
        # Format retrieved content
        retrieved_text = "\n\n---\n\n".join([
            f"[Document {i+1} (score: {r.score:.3f})]\n{r.content}"
            for i, r in enumerate(retrieved)
        ])
        
        # Format prior evidence summary
        prior_summary = ""
        if prior_evidence:
            prior_summary = f"\n\nPRIOR EVIDENCE ALREADY COLLECTED:\n" + "\n---\n".join([
                e[:200] + "..." for e in prior_evidence[-3:]  # Last 3 pieces
            ])
        
        prompt = f"""You are a legal research analyst performing multi-hop document retrieval.

ORIGINAL QUESTION: {original_query}

CURRENT SUB-QUERY: {current_query}

NEWLY RETRIEVED DOCUMENTS:
{retrieved_text}
{prior_summary}

Analyze the retrieved documents and respond in EXACTLY this JSON format:
{{
    "completeness": <float 0.0-1.0 indicating how completely the original question can be answered with all evidence so far>,
    "gaps": [<list of specific information gaps that still need to be filled to fully answer the original question>],
    "new_info": "<summary of genuinely new information from this retrieval hop that wasn't in prior evidence>",
    "reasoning": "<brief explanation of your analysis>"
}}

Be precise about gaps. A gap should be a specific, searchable piece of missing information.
If all Sagaz factors are covered but risk assessment is missing, that's a gap.
If case precedents are cited but their outcomes are unknown, that's a gap.
Respond ONLY with the JSON object."""
        
        try:
            response = self.chat.generate(
                prompt,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1024,
            )
            
            # Parse JSON response
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            # Validate and sanitize
            return {
                "completeness": max(0.0, min(1.0, float(result.get("completeness", 0.0)))),
                "gaps": result.get("gaps", [])[:5],  # Cap at 5 gaps
                "new_info": result.get("new_info", ""),
                "reasoning": result.get("reasoning", ""),
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Gap detection parsing error: {e}")
            return {
                "completeness": 0.5,
                "gaps": ["Unable to parse gap analysis — continuing with general search"],
                "new_info": "",
                "reasoning": f"Parse error: {str(e)[:100]}",
            }
    
    def _should_stop(
        self,
        state: RetrievalState,
        analysis: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Determine whether to stop retrieving or continue with another hop.
        
        This is the most critical research design point — the stopping criterion
        function. The article calls this out specifically:
        "如何设计这个中途停止和继续搜索的判断函数，就是一个非常有价值的研究点"
        
        Stopping conditions (any triggers stop):
        1. Completeness exceeds threshold
        2. No gaps remaining
        3. New information is below minimum threshold
        4. Max hops reached (handled by caller)
        
        Returns:
            Tuple of (should_stop: bool, reason: str)
        """
        # Condition 1: Evidence is sufficiently complete
        if analysis["completeness"] >= self.completeness_threshold:
            return True, f"Completeness ({analysis['completeness']:.2f}) >= threshold ({self.completeness_threshold})"
        
        # Condition 2: No more gaps identified
        if not analysis["gaps"]:
            return True, "No information gaps remaining"
        
        # Condition 3: Diminishing returns — too little new information
        new_info_tokens = len(analysis.get("new_info", "").split())
        if len(state.hops) > 1 and new_info_tokens < self.min_new_info_tokens:
            return True, f"Diminishing returns: only {new_info_tokens} new tokens (min: {self.min_new_info_tokens})"
        
        # Condition 4: Completeness plateaued (no improvement from last hop)
        if len(state.hops) >= 2:
            prev_completeness = state.hops[-2].completeness_score
            curr_completeness = analysis["completeness"]
            if curr_completeness <= prev_completeness + 0.05:
                return True, f"Completeness plateaued ({prev_completeness:.2f} -> {curr_completeness:.2f})"
        
        return False, ""
    
    def _generate_sub_query(
        self,
        gap: str,
        original_query: str,
        prior_evidence: List[str],
    ) -> str:
        """
        Generate a targeted sub-query to fill a specific information gap.
        
        The sub-query should be optimized for vector similarity search,
        meaning it should contain the key legal terms and concepts that
        would appear in documents containing the missing information.
        """
        prior_summary = ""
        if prior_evidence:
            prior_summary = "\n".join([e[:150] + "..." for e in prior_evidence[-2:]])
        
        prompt = f"""Generate a concise, targeted search query to find legal documents that address this information gap.

ORIGINAL QUESTION: {original_query}

INFORMATION GAP TO FILL: {gap}

EVIDENCE ALREADY COLLECTED (summary):
{prior_summary}

Generate a search query that:
1. Uses specific legal terminology that would appear in relevant documents
2. Is focused on the specific gap (not the whole original question)
3. Is 1-2 sentences maximum
4. Would work well for semantic similarity search

Respond with ONLY the search query text, nothing else."""
        
        try:
            sub_query = self.chat.generate(
                prompt,
                temperature=0.3,
                max_tokens=128,
            ).strip()
            
            # Remove quotes if present
            sub_query = sub_query.strip('"').strip("'")
            return sub_query
            
        except Exception as e:
            logger.warning(f"Sub-query generation error: {e}")
            return gap  # Fall back to using the gap description as the query
    
    def _generate_final_answer(
        self,
        question: str,
        evidence: List[str],
        hops: List[RetrievalHop],
    ) -> str:
        """
        Generate the final answer using the complete evidence chain.
        
        Unlike single-hop RAG which just stuffs retrieved docs into a prompt,
        this includes the reasoning trace from multi-hop retrieval to give
        the LLM a structured view of how the evidence was assembled.
        """
        # Format evidence with hop annotations
        evidence_text = ""
        for i, hop in enumerate(hops):
            evidence_text += f"\n\n=== HOP {hop.hop_number} (query: '{hop.query[:60]}...') ===\n"
            for doc in hop.retrieved_docs:
                evidence_text += f"\n[Source, score={doc['score']}]\n{doc['excerpt']}\n"
        
        prompt = f"""You are a legal research assistant for the Deel Lab for Global Employment.

You have conducted a multi-hop retrieval process to answer a complex legal question.
The evidence was gathered iteratively — each hop targeted specific information gaps.

QUESTION: {question}

EVIDENCE CHAIN (assembled through {len(hops)} retrieval hops):
{evidence_text}

Based on this comprehensive evidence chain, provide a thorough legal analysis that:
1. Directly answers the question with specific citations to the evidence
2. Identifies all relevant legal tests and factors mentioned in the evidence
3. Notes the strength of the evidence (what is well-supported vs. uncertain)
4. Highlights any remaining gaps or areas where further research is needed
5. Provides a clear conclusion with appropriate hedging for uncertain areas

Structure your response with clear headings and numbered points."""
        
        system_instruction = """You are a legal research assistant for the Deel Lab for Global Employment.
Your role is to provide accurate, well-cited answers based on multi-hop retrieved evidence.
When evidence from different hops conflicts, note the conflict explicitly.
Always distinguish between what the evidence directly supports and what you are inferring."""
        
        try:
            return self.chat.generate(
                prompt,
                system_instruction=system_instruction,
                temperature=0.4,
                max_tokens=3072,
            )
        except Exception as e:
            logger.error(f"Final answer generation error: {e}")
            return f"Error generating final answer: {str(e)}"
    
    def _build_reasoning_trace(self, state: RetrievalState) -> str:
        """Build a human-readable reasoning trace from the retrieval state."""
        lines = [
            f"Multi-Hop Retrieval Trace for: {state.original_query}",
            f"Status: {state.status}",
            f"Total hops: {len(state.hops)}",
            f"Final completeness: {state.overall_completeness:.2f}",
            "",
        ]
        
        for hop in state.hops:
            lines.append(f"--- Hop {hop.hop_number} ---")
            lines.append(f"  Query: {hop.query}")
            lines.append(f"  Documents retrieved: {len(hop.retrieved_docs)}")
            lines.append(f"  Completeness after hop: {hop.completeness_score:.2f}")
            lines.append(f"  Gaps identified: {hop.identified_gaps}")
            lines.append(f"  Reasoning: {hop.reasoning}")
            lines.append("")
        
        if state.remaining_gaps:
            lines.append(f"Remaining gaps: {state.remaining_gaps}")
        
        return "\n".join(lines)


def main():
    """Test the multi-hop retriever"""
    print("\n" + "=" * 60)
    print("TESTING MULTI-HOP RETRIEVER")
    print("=" * 60)
    
    try:
        retriever = MultiHopRetriever(max_hops=3)
        
        # Test with a complex multi-factor question
        result = retriever.retrieve(
            "What are all the factors that distinguish an employee from an "
            "independent contractor under the Sagaz test, and how do Ontario "
            "courts weigh these factors when the worker uses company equipment "
            "but sets their own hours?",
            top_k_per_hop=3,
        )
        
        print(f"\n📝 Query: {result.query}")
        print(f"\n🔗 Total hops: {result.total_hops}")
        print(f"📊 Final completeness: {result.final_completeness:.2f}")
        print(f"⏱️  Duration: {result.duration_ms:.0f}ms")
        print(f"\n📖 Sources: {len(result.sources)}")
        print(f"\n💬 Answer:\n{result.answer[:500]}...")
        print(f"\n🔍 Reasoning Trace:\n{result.reasoning_trace}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
