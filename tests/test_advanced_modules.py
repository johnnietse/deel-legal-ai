# Tests for Advanced RAG, MCTS Agent, and Evaluation Framework
"""
Test suite covering all 5 new research modules:
1. Multi-Hop Retriever
2. Knowledge Graph + Hybrid Retriever
3. MCTS Legal Reasoning Agent
4. Dynamic Benchmark Generator
5. Debiased LLM Judge
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Module 1: Multi-Hop Retriever Tests
# ============================================================

class TestMultiHopRetriever:
    """Tests for the multi-hop retrieval engine."""
    
    def test_retrieval_state_initialization(self):
        """Test RetrievalState initializes correctly."""
        from rag_pipeline.multi_hop_retriever import RetrievalState
        
        state = RetrievalState(
            original_query="test query",
            current_sub_query="test query",
        )
        
        assert state.original_query == "test query"
        assert state.status == "init"
        assert state.overall_completeness == 0.0
        assert len(state.hops) == 0
        assert len(state.accumulated_evidence) == 0
    
    def test_retrieval_hop_dataclass(self):
        """Test RetrievalHop stores hop information."""
        from rag_pipeline.multi_hop_retriever import RetrievalHop
        
        hop = RetrievalHop(
            hop_number=1,
            query="initial query",
            retrieved_docs=[{"id": "doc1", "score": 0.9}],
            identified_gaps=["missing factor analysis"],
            completeness_score=0.4,
            new_information="Found control factor details",
            reasoning="Need more information on tool ownership",
        )
        
        assert hop.hop_number == 1
        assert len(hop.retrieved_docs) == 1
        assert len(hop.identified_gaps) == 1
        assert hop.completeness_score == 0.4
    
    def test_should_stop_high_completeness(self):
        """Test stopping criterion triggers on high completeness."""
        from rag_pipeline.multi_hop_retriever import MultiHopRetriever, RetrievalState, RetrievalHop
        
        retriever = MultiHopRetriever.__new__(MultiHopRetriever)
        retriever.completeness_threshold = 0.8
        retriever.min_new_info_tokens = 50
        
        state = RetrievalState(
            original_query="test",
            current_sub_query="test",
        )
        state.hops = [
            RetrievalHop(1, "q1", [], [], 0.5, "info", "reason"),
        ]
        
        analysis = {"completeness": 0.85, "gaps": [], "new_info": "some new info " * 20}
        
        should_stop, reason = retriever._should_stop(state, analysis)
        assert should_stop is True
        assert "Completeness" in reason
    
    def test_should_stop_no_gaps(self):
        """Test stopping criterion triggers when no gaps remain."""
        from rag_pipeline.multi_hop_retriever import MultiHopRetriever, RetrievalState
        
        retriever = MultiHopRetriever.__new__(MultiHopRetriever)
        retriever.completeness_threshold = 0.8
        retriever.min_new_info_tokens = 50
        
        state = RetrievalState(original_query="test", current_sub_query="test")
        analysis = {"completeness": 0.5, "gaps": [], "new_info": ""}
        
        should_stop, reason = retriever._should_stop(state, analysis)
        assert should_stop is True
        assert "gaps" in reason.lower()
    
    def test_multi_hop_result_structure(self):
        """Test MultiHopResult has all required fields."""
        from rag_pipeline.multi_hop_retriever import MultiHopResult
        
        result = MultiHopResult(
            query="test",
            answer="answer",
            evidence_chain=[],
            total_hops=3,
            final_completeness=0.85,
            sources=[],
            reasoning_trace="trace",
            duration_ms=1500.0,
        )
        
        assert result.total_hops == 3
        assert result.final_completeness == 0.85


# ============================================================
# Module 2: Knowledge Graph Tests
# ============================================================

class TestKnowledgeGraph:
    """Tests for the legal knowledge graph."""
    
    def test_add_triple(self):
        """Test adding triples to the graph."""
        from rag_pipeline.knowledge_graph import LegalKnowledgeGraph, Triple
        
        kg = LegalKnowledgeGraph()
        
        triple = Triple(
            subject="Sagaz Case",
            subject_type="Case",
            predicate="applies_test",
            object="Sagaz Test",
            object_type="LegalTest",
            confidence=0.95,
        )
        
        kg.add_triple(triple)
        
        assert kg.node_count == 2
        assert kg.edge_count == 1
    
    def test_entity_index(self):
        """Test entity type indexing."""
        from rag_pipeline.knowledge_graph import LegalKnowledgeGraph, Triple
        
        kg = LegalKnowledgeGraph()
        
        kg.add_triple(Triple("Case A", "Case", "cites", "Case B", "Case"))
        kg.add_triple(Triple("Case A", "Case", "applies_test", "Sagaz", "LegalTest"))
        
        assert "Case A" in kg._entity_index["Case"]
        assert "Case B" in kg._entity_index["Case"]
        assert "Sagaz" in kg._entity_index["LegalTest"]
    
    def test_subgraph_query(self):
        """Test subgraph retrieval around an entity."""
        from rag_pipeline.knowledge_graph import LegalKnowledgeGraph, Triple
        
        kg = LegalKnowledgeGraph()
        
        kg.add_triple(Triple("Case A", "Case", "cites", "Case B", "Case"))
        kg.add_triple(Triple("Case B", "Case", "applies_test", "Sagaz", "LegalTest"))
        kg.add_triple(Triple("Sagaz", "LegalTest", "involves_factor", "Control", "Factor"))
        kg.add_triple(Triple("Unrelated", "Case", "cites", "Other", "Case"))
        
        result = kg.query_subgraph("Case B", max_depth=1)
        
        assert result.center_entity == "Case B"
        assert len(result.nodes) >= 2  # At least Case B and its neighbors
        assert len(result.linearized_text) > 0
    
    def test_subgraph_to_text(self):
        """Test graph-to-text linearization."""
        from rag_pipeline.knowledge_graph import LegalKnowledgeGraph, Triple
        
        kg = LegalKnowledgeGraph()
        
        kg.add_triple(Triple("Sagaz", "LegalTest", "involves_factor", "Control", "Factor"))
        kg.add_triple(Triple("Sagaz", "LegalTest", "involves_factor", "Tools", "Factor"))
        
        result = kg.query_subgraph("Sagaz", max_depth=1)
        text = result.linearized_text
        
        assert "Sagaz" in text
        assert "INVOLVES_FACTOR" in text
        assert "Control" in text
    
    def test_save_and_load(self, tmp_path):
        """Test graph persistence."""
        from rag_pipeline.knowledge_graph import LegalKnowledgeGraph, Triple
        
        kg = LegalKnowledgeGraph()
        kg.add_triple(Triple("Case A", "Case", "cites", "Case B", "Case"))
        kg.add_triple(Triple("Case B", "Case", "applies_test", "Test", "LegalTest"))
        
        save_path = tmp_path / "test_graph.json"
        kg.save(str(save_path))
        
        # Load into new graph
        kg2 = LegalKnowledgeGraph()
        kg2.load(str(save_path))
        
        assert kg2.node_count == kg.node_count
        assert kg2.edge_count == kg.edge_count
    
    def test_find_entity_case_insensitive(self):
        """Test entity lookup is case-insensitive."""
        from rag_pipeline.knowledge_graph import LegalKnowledgeGraph, Triple
        
        kg = LegalKnowledgeGraph()
        kg.add_triple(Triple("Sagaz Test", "LegalTest", "involves_factor", "Control", "Factor"))
        
        assert kg._find_entity("sagaz test") == "Sagaz Test"
        assert kg._find_entity("SAGAZ TEST") == "Sagaz Test"
        assert kg._find_entity("sagaz") == "Sagaz Test"  # Substring match


# ============================================================
# Module 3: MCTS Legal Reasoning Agent Tests
# ============================================================

class TestMCTSAgent:
    """Tests for the MCTS-based reasoning agent."""
    
    def test_reasoning_node_ucb1(self):
        """Test UCB1 calculation for node selection."""
        from rag_pipeline.legal_reasoning_agent import ReasoningNode
        
        node = ReasoningNode(
            id="test", hypothesis="test", factor="control",
            visits=10, total_reward=7.0,
        )
        
        # UCB1 = avg_reward + C * sqrt(ln(parent_visits) / visits)
        ucb1 = node.ucb1(parent_visits=100, exploration_constant=1.414)
        
        assert ucb1 > node.average_reward  # UCB1 should be > exploitation alone
        assert node.average_reward == 0.7
    
    def test_reasoning_node_unvisited_infinity(self):
        """Test that unvisited nodes have infinite UCB1 (always explored)."""
        from rag_pipeline.legal_reasoning_agent import ReasoningNode
        
        node = ReasoningNode(
            id="test", hypothesis="test", factor="control",
            visits=0, total_reward=0.0,
        )
        
        assert node.ucb1(parent_visits=100) == float('inf')
    
    def test_sagaz_factors_complete(self):
        """Test that all 5 Sagaz factors are defined."""
        from rag_pipeline.legal_reasoning_agent import SAGAZ_FACTORS
        
        factor_names = [f["name"] for f in SAGAZ_FACTORS]
        
        assert "control" in factor_names
        assert "ownership_of_tools" in factor_names
        assert "chance_of_profit" in factor_names
        assert "risk_of_loss" in factor_names
        assert "integration" in factor_names
        assert len(SAGAZ_FACTORS) == 5
    
    def test_initial_expansion(self):
        """Test that initial expansion creates 2 nodes per Sagaz factor."""
        from rag_pipeline.legal_reasoning_agent import LegalReasoningAgent, ReasoningNode
        
        agent = LegalReasoningAgent.__new__(LegalReasoningAgent)
        agent._nodes = {}
        agent._node_counter = 0
        
        root = ReasoningNode(id="root", hypothesis="classify", factor="root", depth=0)
        agent._nodes[root.id] = root
        
        agent._initial_expand(root, "test facts")
        
        # Should have 10 children: 2 per Sagaz factor (employee + contractor)
        assert len(root.children) == 10
        
        # Check both directions exist for each factor
        factors_with_employee = set()
        factors_with_contractor = set()
        for child in root.children:
            if "EMPLOYEE" in child.hypothesis.upper():
                factors_with_employee.add(child.factor)
            else:
                factors_with_contractor.add(child.factor)
        
        assert len(factors_with_employee) == 5
        assert len(factors_with_contractor) == 5
    
    def test_backpropagation(self):
        """Test MCTS backpropagation updates all ancestors."""
        from rag_pipeline.legal_reasoning_agent import LegalReasoningAgent, ReasoningNode
        
        agent = LegalReasoningAgent.__new__(LegalReasoningAgent)
        agent._nodes = {}
        
        root = ReasoningNode(id="root", hypothesis="root", factor="root", depth=0)
        child = ReasoningNode(id="child", hypothesis="child", factor="control",
                             parent_id="root", depth=1)
        grandchild = ReasoningNode(id="grandchild", hypothesis="gc", factor="control",
                                  parent_id="child", depth=2)
        
        agent._nodes = {"root": root, "child": child, "grandchild": grandchild}
        
        agent._backpropagate(grandchild, reward=0.8)
        
        assert grandchild.visits == 1
        assert grandchild.total_reward == 0.8
        assert child.visits == 1
        assert child.total_reward == 0.8
        assert root.visits == 1
        assert root.total_reward == 0.8


# ============================================================
# Module 4: Dynamic Benchmark Tests
# ============================================================

class TestDynamicBenchmark:
    """Tests for the dynamic benchmark generator."""
    
    def test_generate_case_deterministic(self):
        """Test that same seed produces same case."""
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
        
        gen = LegalBenchmarkGenerator()
        case1 = gen.generate_case(seed=42, difficulty="medium")
        case2 = gen.generate_case(seed=42, difficulty="medium")
        
        assert case1.scenario == case2.scenario
        assert case1.expected_classification == case2.expected_classification
        assert case1.case_id == case2.case_id
    
    def test_generate_case_unique(self):
        """Test that different seeds produce different cases."""
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
        
        gen = LegalBenchmarkGenerator()
        case1 = gen.generate_case(seed=42, difficulty="medium")
        case2 = gen.generate_case(seed=43, difficulty="medium")
        
        assert case1.scenario != case2.scenario
    
    def test_difficulty_levels(self):
        """Test that all difficulty levels generate valid cases."""
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
        
        gen = LegalBenchmarkGenerator()
        
        for difficulty in ["easy", "medium", "hard"]:
            case = gen.generate_case(seed=42, difficulty=difficulty)
            
            assert case.difficulty == difficulty
            assert len(case.scenario) > 50
            assert case.expected_classification in ["Employee", "Independent Contractor"]
            assert len(case.expected_factors) == 5
            assert len(case.evaluation_rubric) == 6
    
    def test_generate_suite(self):
        """Test suite generation with correct distribution."""
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
        
        gen = LegalBenchmarkGenerator()
        suite = gen.generate_suite(n_cases=10)
        
        assert suite.n_cases == 10
        assert len(suite.cases) == 10
        assert suite.suite_id is not None
        assert suite.generated_at is not None
        
        # Check difficulty distribution exists
        total_in_dist = sum(suite.difficulty_distribution.values())
        assert total_in_dist == 10
    
    def test_case_has_all_rubric_dimensions(self):
        """Test that generated cases have rubrics for all 6 evaluation dimensions."""
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
        from config import EVAL_DIMENSIONS
        
        gen = LegalBenchmarkGenerator()
        case = gen.generate_case(seed=42, difficulty="medium")
        
        for dim in EVAL_DIMENSIONS:
            assert dim in case.evaluation_rubric, f"Missing rubric for {dim}"
    
    def test_suite_save_and_load(self, tmp_path):
        """Test suite serialization."""
        from evaluation.dynamic_benchmark import LegalBenchmarkGenerator
        
        gen = LegalBenchmarkGenerator()
        suite = gen.generate_suite(n_cases=5)
        
        save_path = tmp_path / "test_suite.json"
        suite.save(str(save_path))
        
        assert save_path.exists()
        
        with open(save_path) as f:
            data = json.load(f)
        
        assert data["n_cases"] == 5
        assert len(data["cases"]) == 5


# ============================================================
# Module 5: LLM Judge Tests
# ============================================================

class TestDebiasedJudge:
    """Tests for the debiased LLM judge."""
    
    def test_legal_components_weights_sum_to_one(self):
        """Test that component weights sum to 1.0."""
        from evaluation.llm_judge import DebiasedLegalJudge
        
        total_weight = sum(
            info["weight"] for info in DebiasedLegalJudge.LEGAL_COMPONENTS.values()
        )
        assert abs(total_weight - 1.0) < 0.001
    
    def test_legal_components_complete(self):
        """Test all 5 evaluation components are defined."""
        from evaluation.llm_judge import DebiasedLegalJudge
        
        expected = [
            "argument_identification",
            "reasoning_chain",
            "evidence_usage",
            "legal_accuracy",
            "practical_utility",
        ]
        
        for comp in expected:
            assert comp in DebiasedLegalJudge.LEGAL_COMPONENTS
    
    def test_judge_result_structure(self):
        """Test JudgeResult dataclass."""
        from evaluation.llm_judge import JudgeResult
        
        result = JudgeResult(
            overall_score=0.75,
            component_scores={"arg": 0.8, "chain": 0.7},
            feedback="Good analysis",
            debiasing_applied=["rubric_decomposition"],
        )
        
        assert result.overall_score == 0.75
        assert "rubric_decomposition" in result.debiasing_applied
        
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["overall_score"] == 0.75
    
    def test_pairwise_result_structure(self):
        """Test PairwiseResult dataclass."""
        from evaluation.llm_judge import PairwiseResult
        
        result = PairwiseResult(
            winner="A",
            score_a=0.8,
            score_b=0.6,
            confidence=0.7,
            position_bias_detected=False,
        )
        
        assert result.winner == "A"
        assert not result.position_bias_detected


# ============================================================
# Module 4b: Bias Detector Tests
# ============================================================

class TestBiasDetector:
    """Tests for the bias detection module."""
    
    def test_bias_metric_structure(self):
        """Test BiasMetric dataclass."""
        from evaluation.bias_detector import BiasMetric
        
        metric = BiasMetric(
            bias_type="position",
            magnitude=0.15,
            direction="favors_first",
            confidence=0.8,
            details="Score difference of 0.15",
        )
        
        assert metric.bias_type == "position"
        assert metric.magnitude == 0.15
    
    def test_bias_report_summary(self):
        """Test BiasReport generates readable summary."""
        from evaluation.bias_detector import BiasReport, BiasMetric
        
        report = BiasReport(
            overall_bias_score=0.2,
            biases_detected=[
                BiasMetric("position", 0.1, "favors_first", 0.9, "Minor position bias"),
                BiasMetric("length", 0.3, "favors_longer", 0.8, "Moderate length bias"),
            ],
            recommendations=["Use position swapping"],
            test_sample_size=10,
        )
        
        summary = report.summary()
        
        assert "BIAS DETECTION REPORT" in summary
        assert "position" in summary
        assert "length" in summary
    
    def test_detector_without_judge(self):
        """Test bias detector handles missing judge gracefully."""
        from evaluation.bias_detector import BiasDetector
        
        detector = BiasDetector(judge=None)
        
        metric = detector.detect_position_bias("test question", "test response")
        assert metric.magnitude == 0.0
        assert metric.confidence == 0.0


# ============================================================
# Integration Tests
# ============================================================

class TestRAGQueryExtensions:
    """Test that rag_query.py extensions work correctly."""
    
    def test_complexity_estimation_simple(self):
        """Test simple questions are classified as simple."""
        from rag_pipeline.rag_query import LegalRAGQuery
        
        rag = LegalRAGQuery.__new__(LegalRAGQuery)
        
        assert rag._estimate_complexity("What is the Sagaz test?") == "simple"
        assert rag._estimate_complexity("Is this worker an employee?") == "simple"
    
    def test_complexity_estimation_complex(self):
        """Test complex questions are classified as complex."""
        from rag_pipeline.rag_query import LegalRAGQuery
        
        rag = LegalRAGQuery.__new__(LegalRAGQuery)
        
        complex_q = (
            "What are all the factors that distinguish an employee from an "
            "independent contractor and also how does the integration test "
            "interact with the control test across different jurisdictions?"
        )
        
        assert rag._estimate_complexity(complex_q) == "complex"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
