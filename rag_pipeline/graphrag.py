# RAG Pipeline - GraphRAG / LightRAG-style Retrieval
"""
GraphRAG-style retrieval over the existing LegalKnowledgeGraph.

Flow (RAGFlow/LightRAG inspired):
  1. Extract legal entities from the question (reuses
     LegalKnowledgeGraph.extract_entities_from_query)
  2. Traverse the knowledge graph from each entity (BFS subgraph)
  3. Score graph nodes: PageRank x query-similarity (token overlap proxy)
  4. Hybrid retrieval (BM25 + vector) for document passages
  5. Merge graph context + passage context, generate the answer
  6. Return a RAGResponse-compatible structure so rag_query.query_smart
     can route to this mode

Config-gated by GRAPHRAG_ENABLED. The module imports cleanly even when
disabled; query_graphrag raises RuntimeError with a clear message.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from rag_pipeline.knowledge_graph import LegalKnowledgeGraph
from rag_pipeline.rag_query import RAGResponse

# Setup logging
logger = logging.getLogger(__name__)


def compute_pagerank(
    kg: LegalKnowledgeGraph,
    damping: Optional[float] = None,
) -> Dict[str, float]:
    """Compute PageRank scores for every node in the knowledge graph.

    Uses networkx.pagerank over kg.graph (a DiGraph). Returns an empty
    dict when the graph is empty or networkx is unavailable.

    Args:
        kg: Loaded LegalKnowledgeGraph instance.
        damping: PageRank damping factor (defaults to
            config.GRAPHRAG_PAGERANK_DAMPING).

    Returns:
        Mapping of node name -> PageRank score.
    """
    alpha = damping if damping is not None else config.GRAPHRAG_PAGERANK_DAMPING

    if kg.graph is None or kg.graph.number_of_nodes() == 0:
        return {}

    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx unavailable for PageRank — returning empty scores")
        return {}

    try:
        return dict(nx.pagerank(kg.graph, alpha=alpha))
    except Exception as e:
        logger.warning(f"PageRank computation failed: {e}")
        return {}


def extract_entities(kg: LegalKnowledgeGraph, text: str) -> List[str]:
    """Extract legal entity names from text using the existing KG extractor."""
    return kg.extract_entities_from_query(text)


def _token_overlap(query: str, text: str) -> float:
    """Simple token-overlap similarity proxy (query tokens in text)."""
    q_tokens = set(query.lower().split())
    if not q_tokens:
        return 0.0
    t_tokens = set(text.lower().split())
    if not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def _score_graph_nodes(
    kg: LegalKnowledgeGraph,
    nodes: List[Dict[str, Any]],
    question: str,
) -> Dict[str, float]:
    """Score graph nodes as PageRank x query-similarity.

    A small floor keeps pure-PageRank signal alive when no token overlap
    exists between the question and a node's label/type.
    """
    pagerank = compute_pagerank(kg)
    scores: Dict[str, float] = {}
    for node in nodes:
        node_id = node.get("id", "")
        if not node_id:
            continue
        node_text = " ".join(str(v) for v in node.values())
        overlap = _token_overlap(question, node_text)
        scores[node_id] = pagerank.get(node_id, 0.0) * (0.1 + overlap)
    return scores


def _build_passages_text(results) -> str:
    """Render hybrid retrieval results as a linear passage block."""
    if not results:
        return "No document passages retrieved."
    lines = []
    for i, r in enumerate(results):
        case_name = (r.metadata or {}).get("title", (r.metadata or {}).get("case_name", "Unknown"))
        lines.append(f"[Passage {i + 1}, Score: {r.score:.3f}, Case: {case_name}]")
        lines.append(r.content)
    return "\n\n".join(lines)


def _generate_graphrag_answer(rag_query, question: str, graph_text: str, passages_text: str) -> str:
    """Generate an answer using merged graph + passage context."""
    prompt = f"""You are a legal research assistant with access to both document passages
and a knowledge graph of legal relationships.

QUESTION: {question}

=== RETRIEVED DOCUMENT PASSAGES ===
{passages_text}

=== KNOWLEDGE GRAPH CONTEXT ===
{graph_text}

Using BOTH the document passages AND the knowledge graph relationships, provide a
comprehensive legal analysis. When the knowledge graph shows citation chains or
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

    return rag_query.chat.generate(
        prompt,
        system_instruction=system_instruction,
        temperature=0.4,
        max_tokens=2048,
    )


def query_graphrag(
    rag_query,
    question: str,
    top_k: int = 5,
    namespace: str = "",
    filter: Optional[Dict[str, Any]] = None,
    verify: bool = False,
    user_id: Optional[str] = None,
) -> RAGResponse:
    """GraphRAG query mode: PageRank-scored subgraphs + hybrid passages.

    Args:
        rag_query: LegalRAGQuery instance providing chat, hybrid_retriever,
            and _format_sources.
        question: Legal question to answer.
        top_k: Number of document sources to return.
        namespace: Vector store namespace.
        filter: Metadata filter for hybrid retrieval.
        verify: Run post-hoc verification on the generated answer.
        user_id: Optional user id for metrics.

    Returns:
        RAGResponse with retrieval_mode='graphrag'. Falls back to plain
        rag_query.query(...) when no entities or no graph nodes are found.

    Raises:
        RuntimeError: When GRAPHRAG_ENABLED is False.
    """
    if not config.GRAPHRAG_ENABLED:
        raise RuntimeError(
            "GraphRAG disabled: set GRAPHRAG_ENABLED=1 in the environment to enable graphrag query mode."
        )

    # Load the existing knowledge graph (no-op gracefully when file missing)
    kg = LegalKnowledgeGraph()
    kg.load()

    # --- (a) Extract entities from the question ---
    entities = extract_entities(kg, question)[: config.GRAPHRAG_TOP_ENTITIES]
    if not entities:
        logger.info("GraphRAG: no entities found — falling back to plain query")
        return rag_query.query(
            question, top_k=top_k, namespace=namespace,
            filter=filter, verify=verify, user_id=user_id,
        )

    # --- (b) Traverse subgraphs for each entity ---
    graph_texts: List[str] = []
    graph_nodes: List[Dict[str, Any]] = []
    seen_entities = set()
    for entity in entities:
        entity_key = entity.lower()
        if entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        try:
            subgraph = kg.query_subgraph(entity, max_depth=config.GRAPHRAG_MAX_DEPTH)
        except Exception as e:
            logger.warning(f"GraphRAG: subgraph query failed for '{entity}': {e}")
            continue
        if subgraph.nodes:
            graph_texts.append(subgraph.linearized_text)
            graph_nodes.extend(subgraph.nodes)

    if not graph_nodes:
        logger.info("GraphRAG: entities found but no graph nodes — falling back to plain query")
        return rag_query.query(
            question, top_k=top_k, namespace=namespace,
            filter=filter, verify=verify, user_id=user_id,
        )

    # --- (c) Score graph nodes: PageRank x query-similarity ---
    node_scores = _score_graph_nodes(kg, graph_nodes, question)
    top_scored = sorted(node_scores, key=node_scores.get, reverse=True)[:config.GRAPHRAG_MERGE_TOP_K]
    pagerank_note = (
        "Most relevant graph entities (by PageRank x query similarity): "
        + ", ".join(top_scored)
    ) if top_scored else ""

    # --- (d) Hybrid retrieval ---
    try:
        hybrid_results = rag_query.hybrid_retriever.retrieve(
            query=question,
            top_k=config.GRAPHRAG_MERGE_TOP_K,
            namespace=namespace,
            filter=filter,
        )
    except Exception as e:
        logger.error(f"GraphRAG: hybrid retrieval failed: {e}")
        return rag_query.query(
            question, top_k=top_k, namespace=namespace,
            filter=filter, verify=verify, user_id=user_id,
        )

    sources = rag_query._format_sources(hybrid_results)

    # --- (e) Merge context and generate ---
    graph_text = "\n\n".join(graph_texts)
    if pagerank_note:
        graph_text = f"{graph_text}\n\n{pagerank_note}"
    passages_text = _build_passages_text(hybrid_results)

    confidence = "low"
    if hybrid_results:
        avg_score = sum(r.score for r in hybrid_results) / len(hybrid_results)
        confidence = "high" if avg_score > 0.8 else "medium" if avg_score > 0.6 else "low"

    try:
        answer = _generate_graphrag_answer(rag_query, question, graph_text, passages_text)
    except Exception as e:
        logger.error(f"GraphRAG: answer generation failed: {e}")
        return RAGResponse(
            query=question,
            answer="I found relevant legal sources and graph relationships, but encountered an error generating the full answer. Please review the sources below.",
            sources=sources,
            confidence=confidence,
            retrieval_mode="graphrag",
            metrics={"pagerank_nodes": len(graph_nodes), "entities": entities},
            status="degraded",
            error_type="generation_failed",
            error_message=str(e),
        )

    response = RAGResponse(
        query=question,
        answer=answer,
        sources=sources,
        confidence=confidence,
        retrieval_mode="graphrag",
        metrics={"pagerank_nodes": len(graph_nodes), "entities": entities},
        status="ok",
    )

    # --- (f) Optional post-hoc verification ---
    if verify:
        try:
            verification = rag_query.verifier.verify_grounding(answer, sources)
            response.verification = verification.to_dict()
            if not verification.is_grounded:
                if verification.corrected_answer:
                    response.answer = (
                        f"{verification.corrected_answer}\n\n"
                        "[System Note: The original response was modified because it "
                        "contained claims unsupported by the retrieved documents.]"
                    )
                elif verification.unsupported_claims:
                    response.answer = (
                        f"{answer}\n\n"
                        "[System Warning: The verification module flagged the following "
                        "claims as potentially unsupported: "
                        f"{', '.join(verification.unsupported_claims)}]"
                    )
        except Exception as e:
            logger.warning(f"GraphRAG: verification skipped (error): {e}")

    return response


def main():
    """Smoke test for the GraphRAG module."""
    print("\n" + "=" * 60)
    print("TESTING GRAPHRAG MODULE")
    print("=" * 60)

    from rag_pipeline.rag_query import LegalRAGQuery

    kg = LegalKnowledgeGraph()
    kg.load()
    print(f"Graph: {kg.node_count} nodes, {kg.edge_count} edges")

    pagerank = compute_pagerank(kg)
    print(f"PageRank nodes: {len(pagerank)}")

    if not config.GRAPHRAG_ENABLED:
        print("GRAPHRAG_ENABLED is False — query_graphrag will raise RuntimeError (expected).")
        try:
            query_graphrag(None, "test")
        except RuntimeError as e:
            print(f"  RuntimeError raised as expected: {e}")
        return

    rag = LegalRAGQuery()
    response = query_graphrag(rag, "How does the Sagaz test apply to worker classification in Ontario?")
    print(f"\nRetrieval mode: {response.retrieval_mode}")
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {len(response.sources)}")
    print(f"Metrics: {response.metrics}")
    print(f"\nAnswer:\n{response.answer[:500]}...")


if __name__ == "__main__":
    main()
