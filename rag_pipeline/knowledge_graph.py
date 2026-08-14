# RAG Pipeline - Legal Knowledge Graph
"""
Knowledge graph construction and traversal for legal document retrieval.

Instead of retrieving isolated text chunks, this module builds and queries
a graph of legal entities and their relationships:

- Cases → cites → Cases (precedent chains)
- Cases → applies → LegalTests (e.g., Sagaz test)
- Cases → involves → Factors (control, tools, profit/loss, integration)
- Factors → supports → Classification (employee/contractor)

Research challenge from the article:
"如何把子图的拓扑结构转化为大模型能理解的线性输入"
(How to convert subgraph topology into linear input the LLM can understand)

Key contributions:
- LLM-based entity and relation extraction from legal text
- Subgraph retrieval via entity-seeded BFS traversal
- Graph-to-text linearization preserving topological structure
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.embeddings import GeminiChat
from rag_pipeline.mcts_reasoner import _extract_json_obj, extract_json_array
from config import (
    KNOWLEDGE_GRAPH_PATH,
    KG_MAX_SUBGRAPH_DEPTH,
    KG_ENTITY_TYPES,
    KG_RELATION_TYPES,
)

# Setup logging
logger = logging.getLogger(__name__)

# Try to import networkx, fall back to built-in graph
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("networkx not installed — using built-in graph implementation")


@dataclass
class Triple:
    """A (subject, predicate, object) triple in the knowledge graph"""
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float = 1.0
    source_document: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubgraphResult:
    """Result of a subgraph query"""
    center_entity: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    linearized_text: str
    depth: int


class LegalKnowledgeGraph:
    """
    Legal domain knowledge graph for structured retrieval.
    
    Uses NetworkX for graph operations when available, falls back to
    a simple adjacency list implementation otherwise.
    
    The graph stores:
    - Nodes: Legal entities (cases, courts, tests, factors, etc.)
    - Edges: Legal relationships (cites, applies, involves, etc.)
    - Node attributes: entity_type, metadata
    - Edge attributes: relation_type, confidence, source_document
    """
    
    def __init__(self, chat: Optional[GeminiChat] = None):
        self.chat = chat or GeminiChat()
        
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
        else:
            # Fallback: adjacency list
            self._nodes: Dict[str, Dict[str, Any]] = {}
            self._edges: List[Dict[str, Any]] = []
            self._adjacency: Dict[str, List[str]] = defaultdict(list)
            self._reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
            self.graph = None
        
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)
    
    @property
    def node_count(self) -> int:
        if HAS_NETWORKX:
            return self.graph.number_of_nodes()
        return len(self._nodes)
    
    @property
    def edge_count(self) -> int:
        if HAS_NETWORKX:
            return self.graph.number_of_edges()
        return len(self._edges)
    
    def add_triple(self, triple: Triple):
        """Add a triple (entity-relation-entity) to the graph."""
        if HAS_NETWORKX:
            # Add nodes with type attribute
            self.graph.add_node(triple.subject, entity_type=triple.subject_type)
            self.graph.add_node(triple.object, entity_type=triple.object_type)
            
            # Add edge
            self.graph.add_edge(
                triple.subject,
                triple.object,
                relation=triple.predicate,
                confidence=triple.confidence,
                source_doc=triple.source_document,
            )
        else:
            self._nodes[triple.subject] = {"entity_type": triple.subject_type}
            self._nodes[triple.object] = {"entity_type": triple.object_type}
            self._edges.append({
                "from": triple.subject,
                "to": triple.object,
                "relation": triple.predicate,
                "confidence": triple.confidence,
                "source_doc": triple.source_document,
            })
            self._adjacency[triple.subject].append(triple.object)
            self._reverse_adjacency[triple.object].append(triple.subject)
            self.graph = None  # Keep fallback marker
        
        # Update entity index
        self._entity_index[triple.subject_type].add(triple.subject)
        self._entity_index[triple.object_type].add(triple.object)
    
    def extract_triples_from_text(
        self,
        text: str,
        document_id: str = "",
    ) -> List[Triple]:
        """
        Use Gemini to extract (entity, relation, entity) triples from legal text.
        
        This is the core extraction function — quality here directly determines
        graph quality. Uses structured prompting with explicit entity and
        relation type constraints.
        """
        entity_types_str = ", ".join(KG_ENTITY_TYPES)
        relation_types_str = ", ".join(KG_RELATION_TYPES)
        
        prompt = f"""You are a legal knowledge extraction system. Output ONLY valid JSON.

TEXT:
{text[:3000]}

ENTITY TYPES: {entity_types_str}
RELATION TYPES: {relation_types_str}

Extract triples as JSON:
{{
  "triples": [
    {{"subject": "string", "subject_type": "string", "predicate": "string", "object": "string", "object_type": "string", "confidence": 0.0}}
  ]
}}

RULES:
- ONLY use entity types: {entity_types_str}
- ONLY use relation types: {relation_types_str}
- Output ONLY the JSON object. NO text, NO markdown, NO reasoning.
- If no triples found, return {{"triples": []}}"""
        
        try:
            response = self.chat.generate(
                prompt,
                temperature=0.1,
                max_tokens=2048,
            )
            
            # Parse response — robust extraction handles markdown fences,
            # surrounding prose, and truncated JSON objects.
            data = _extract_json_obj(response)
            if data is None:
                logger.warning(f"Triple extraction failed to parse JSON for {document_id}")
                return []
            triples = []
            
            for t in data.get("triples", []):
                # Validate entity and relation types
                if (t.get("subject_type") in KG_ENTITY_TYPES and
                    t.get("object_type") in KG_ENTITY_TYPES and
                    t.get("predicate") in KG_RELATION_TYPES):
                    
                    triples.append(Triple(
                        subject=t["subject"],
                        subject_type=t["subject_type"],
                        predicate=t["predicate"],
                        object=t["object"],
                        object_type=t["object_type"],
                        confidence=float(t.get("confidence", 0.8)),
                        source_document=document_id,
                    ))
            
            logger.info(f"Extracted {len(triples)} triples from {document_id}")
            return triples
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Triple extraction error for {document_id}: {e}")
            return []
    
    def build_from_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "content",
        id_key: str = "chunk_id",
    ):
        """
        Build the knowledge graph from a list of document chunks.
        
        Processes each document through LLM-based extraction and adds
        all resulting triples to the graph.
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents...")
        
        total_triples = 0
        for i, doc in enumerate(documents):
            text = doc.get(text_key, "")
            doc_id = doc.get(id_key, f"doc_{i}")
            
            if not text.strip():
                continue
            
            triples = self.extract_triples_from_text(text, doc_id)
            
            for triple in triples:
                self.add_triple(triple)
                total_triples += 1
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(documents)} documents, {total_triples} triples so far")
        
        logger.info(f"Knowledge graph built: {self.node_count} nodes, {self.edge_count} edges")
    
    def query_subgraph(
        self,
        entity: str,
        max_depth: int = KG_MAX_SUBGRAPH_DEPTH,
    ) -> SubgraphResult:
        """
        Retrieve connected subgraph around an entity via BFS traversal.
        
        Returns nodes and edges within max_depth hops of the center entity.
        """
        # Find the entity in the graph (case-insensitive fuzzy match)
        matched_entity = self._find_entity(entity)
        
        if not matched_entity:
            return SubgraphResult(
                center_entity=entity,
                nodes=[],
                edges=[],
                linearized_text=f"No entity '{entity}' found in knowledge graph.",
                depth=0,
            )
        
        # BFS traversal
        if HAS_NETWORKX:
            return self._query_subgraph_networkx(matched_entity, max_depth)
        else:
            return self._query_subgraph_builtin(matched_entity, max_depth)
    
    def _find_entity(self, query: str) -> Optional[str]:
        """Find entity by case-insensitive substring match."""
        query_lower = query.lower()
        
        # First try exact match
        all_nodes = list(self.graph.nodes()) if HAS_NETWORKX else list(self._nodes.keys())
        
        for node in all_nodes:
            if node.lower() == query_lower:
                return node
        
        # Then try substring match
        for node in all_nodes:
            if query_lower in node.lower() or node.lower() in query_lower:
                return node
        
        return None
    
    def _query_subgraph_networkx(self, entity: str, max_depth: int) -> SubgraphResult:
        """Query subgraph using NetworkX BFS."""
        # Get subgraph via BFS
        visited = set()
        frontier = {entity}
        
        for depth in range(max_depth):
            next_frontier = set()
            for node in frontier:
                if node not in visited:
                    visited.add(node)
                    # Add successors and predecessors
                    next_frontier.update(self.graph.successors(node))
                    next_frontier.update(self.graph.predecessors(node))
            frontier = next_frontier - visited
            
            if not frontier:
                break
        
        visited.update(frontier)
        
        # Build subgraph
        subgraph = self.graph.subgraph(visited)
        
        nodes = [{
            "id": n,
            **self.graph.nodes[n],
        } for n in subgraph.nodes()]
        
        edges = [{
            "source": u,
            "target": v,
            **d,
        } for u, v, d in subgraph.edges(data=True)]
        
        # Linearize
        linearized = self.subgraph_to_text(nodes, edges, entity)
        
        return SubgraphResult(
            center_entity=entity,
            nodes=nodes,
            edges=edges,
            linearized_text=linearized,
            depth=max_depth,
        )
    
    def _query_subgraph_builtin(self, entity: str, max_depth: int) -> SubgraphResult:
        """Query subgraph using built-in adjacency list."""
        visited = set()
        frontier = {entity}
        
        for depth in range(max_depth):
            next_frontier = set()
            for node in frontier:
                if node not in visited:
                    visited.add(node)
                    next_frontier.update(self._adjacency.get(node, []))
                    next_frontier.update(self._reverse_adjacency.get(node, []))
            frontier = next_frontier - visited
            
            if not frontier:
                break
        
        visited.update(frontier)
        
        nodes = [{
            "id": n,
            **self._nodes.get(n, {}),
        } for n in visited]
        
        edges = [e for e in self._edges
                 if e["from"] in visited and e["to"] in visited]
        
        linearized = self.subgraph_to_text(nodes, edges, entity)
        
        return SubgraphResult(
            center_entity=entity,
            nodes=nodes,
            edges=edges,
            linearized_text=linearized,
            depth=max_depth,
        )
    
    def subgraph_to_text(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        center: str,
    ) -> str:
        """
        Convert subgraph topology into linear text for LLM consumption.
        
        This is the key research question:
        "如何把子图的拓扑结构转化为大模型能理解的线性输入"
        
        Strategy: Hierarchical linearization from the center entity outward,
        organized by relationship type, with explicit structural markers.
        """
        if not nodes:
            return "No graph data available."
        
        # Group edges by relation type
        edges_by_relation: Dict[str, List[Dict]] = defaultdict(list)
        for edge in edges:
            edges_by_relation[edge.get("relation", "related_to")].append(edge)
        
        # Build linearized text
        lines = [f"Knowledge Graph centered on: {center}"]
        lines.append(f"({len(nodes)} entities, {len(edges)} relationships)\n")
        
        # Entity summary
        entity_types = defaultdict(list)
        for node in nodes:
            entity_types[node.get("entity_type", "Unknown")].append(node["id"])
        
        lines.append("ENTITIES:")
        for etype, entities in sorted(entity_types.items()):
            lines.append(f"  [{etype}]: {', '.join(entities[:10])}")
        
        lines.append("\nRELATIONSHIPS:")
        for relation, rel_edges in sorted(edges_by_relation.items()):
            lines.append(f"\n  {relation.upper()}:")
            for edge in rel_edges:
                conf = edge.get("confidence", "")
                conf_str = f" (confidence: {conf:.2f})" if isinstance(conf, (int, float)) else ""
                src = edge.get('source', edge.get('from', '?'))
                tgt = edge.get('target', edge.get('to', '?'))
                lines.append(f"    {src} → {tgt}{conf_str}")
        
        # Generate a narrative summary for the LLM
        lines.append("\nNARRATIVE SUMMARY:")
        
        # Precedent chains
        cites_edges = edges_by_relation.get("cites", [])
        if cites_edges:
            chain = [e.get("source", e.get("from", "?")) for e in cites_edges[:5]]
            chain.append(cites_edges[-1].get("target", cites_edges[-1].get("to", "?")))
            lines.append("  Precedent chain: " + " → ".join(chain))

        # Applied tests
        test_edges = edges_by_relation.get("applies_test", [])
        if test_edges:
            tests = set(e.get("target", e.get("to", "?")) for e in test_edges)
            lines.append(f"  Legal tests applied: {', '.join(tests)}")

        # Factors involved
        factor_edges = edges_by_relation.get("involves_factor", [])
        if factor_edges:
            factors = set(e.get("target", e.get("to", "?")) for e in factor_edges)
            lines.append(f"  Factors considered: {', '.join(factors)}")
        
        return "\n".join(lines)
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entity names from a natural language query using Gemini."""
        prompt = f"""Extract legal entity names from the following query.
Return entities that could be found in a legal knowledge graph.

QUERY: {query}

Return a JSON array of entity name strings:
["entity1", "entity2", ...]

Only include specific named entities (case names, legal tests, statutes, courts).
Do NOT include generic terms like "worker" or "employer".
Respond with ONLY the JSON array."""
        
        try:
            response = self.chat.generate(prompt, temperature=0.1, max_tokens=256)
            parsed = extract_json_array(response)
            return [e for e in parsed if isinstance(e, str)]
        except Exception as e:
            logger.warning(f"Entity extraction error: {e}")
            return []
    
    def save(self, path: Optional[str] = None):
        """Serialize graph to JSON for persistence."""
        path = Path(path or KNOWLEDGE_GRAPH_PATH)
        
        if HAS_NETWORKX:
            nodes_data = []
            for n in self.graph.nodes():
                node_dict = {"id": n}
                node_dict.update(self.graph.nodes[n])
                nodes_data.append(node_dict)
            
            edges_data = []
            for u, v, d in self.graph.edges(data=True):
                edge_dict = {"from": u, "to": v}
                edge_dict.update(d)
                edges_data.append(edge_dict)
            
            data = {"nodes": nodes_data, "edges": edges_data}
        else:
            data = {
                "nodes": [{"id": k, **v} for k, v in self._nodes.items()],
                "edges": self._edges,
            }
        
        data["metadata"] = {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": {k: len(v) for k, v in self._entity_index.items()},
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge graph saved to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load graph from JSON."""
        path = Path(path or KNOWLEDGE_GRAPH_PATH)
        
        if not path.exists():
            logger.warning(f"Knowledge graph file not found: {path}")
            return
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Rebuild graph from nodes
        for node in data.get("nodes", []):
            node = dict(node)  # Copy to avoid mutating loaded data
            node_id = node.pop("id")
            if HAS_NETWORKX:
                self.graph.add_node(node_id, **node)
            else:
                self._nodes[node_id] = node
            
            entity_type = node.get("entity_type", "Unknown")
            self._entity_index[entity_type].add(node_id)
        
        # Rebuild graph from edges
        for edge in data.get("edges", []):
            edge = dict(edge)  # Copy to avoid mutating loaded data
            if HAS_NETWORKX:
                source = edge.pop("from")
                target = edge.pop("to")
                self.graph.add_edge(source, target, **edge)
            else:
                self._edges.append(edge)
                self._adjacency[edge["from"]].append(edge["to"])
                self._reverse_adjacency[edge["to"]].append(edge["from"])
        
        logger.info(f"Knowledge graph loaded: {self.node_count} nodes, {self.edge_count} edges")


def main():
    """Test the knowledge graph module"""
    print("\n" + "=" * 60)
    print("TESTING LEGAL KNOWLEDGE GRAPH")
    print("=" * 60)
    
    kg = LegalKnowledgeGraph()
    
    # Add some sample triples
    sample_triples = [
        Triple("Sagaz Industries v. 671122 Ontario", "Case", "applies_test", "Sagaz Test", "LegalTest"),
        Triple("Sagaz Test", "LegalTest", "involves_factor", "Control over work", "Factor"),
        Triple("Sagaz Test", "LegalTest", "involves_factor", "Ownership of tools", "Factor"),
        Triple("Sagaz Test", "LegalTest", "involves_factor", "Chance of profit", "Factor"),
        Triple("Sagaz Test", "LegalTest", "involves_factor", "Risk of loss", "Factor"),
        Triple("Sagaz Test", "LegalTest", "involves_factor", "Integration into business", "Factor"),
        Triple("Wiebe Door v. MNR", "Case", "cites", "Sagaz Industries v. 671122 Ontario", "Case"),
        Triple("Control over work", "Factor", "supports_classification", "Employee", "Factor"),
    ]
    
    for t in sample_triples:
        kg.add_triple(t)
    
    print(f"\nGraph: {kg.node_count} nodes, {kg.edge_count} edges")
    
    # Query subgraph
    result = kg.query_subgraph("Sagaz Test", max_depth=2)
    print(f"\nSubgraph around 'Sagaz Test':")
    print(f"  Nodes: {len(result.nodes)}")
    print(f"  Edges: {len(result.edges)}")
    print(f"\nLinearized text:\n{result.linearized_text}")
    
    # Test save/load
    kg.save()
    print(f"\n✅ Graph saved to {KNOWLEDGE_GRAPH_PATH}")
    
    kg2 = LegalKnowledgeGraph()
    kg2.load()
    print(f"✅ Graph loaded: {kg2.node_count} nodes, {kg2.edge_count} edges")


if __name__ == "__main__":
    main()
