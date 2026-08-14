# Tests for the General MCTS Reasoning Engine
import pytest
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestEvidenceHeuristic:
    def test_empty_evidence_zero(self):
        from rag_pipeline.mcts_reasoner import evidence_heuristic
        assert evidence_heuristic([], "test question") == 0.0

    def test_coverage_raises_score(self):
        from rag_pipeline.mcts_reasoner import evidence_heuristic
        ev = ["A worker is an employee when control is high."]
        high = evidence_heuristic(ev, "what is an employee under control test?")
        assert 0.0 < high <= 1.0

    def test_volume_bonus(self):
        from rag_pipeline.mcts_reasoner import evidence_heuristic
        ev = ["one", "two", "three"]
        assert evidence_heuristic(ev, "question") > evidence_heuristic([ev[0]], "question")


class TestParseJudgeScores:
    def test_valid_json(self):
        from rag_pipeline.mcts_reasoner import parse_judge_scores
        out = parse_judge_scores(
            '{"precedent_alignment": 0.8, "factor_completeness": 0.7, '
            '"logical_consistency": 0.9, "evidence_strength": 0.6}'
        )
        assert out is not None
        assert out["precedent_alignment"] == 0.8

    def test_fenced_json(self):
        from rag_pipeline.mcts_reasoner import parse_judge_scores
        out = parse_judge_scores(
            '```json\n{"precedent_alignment": 0.5}\n```'
        )
        assert out is not None
        assert out["precedent_alignment"] == 0.5

    def test_invalid_returns_none(self):
        from rag_pipeline.mcts_reasoner import parse_judge_scores
        assert parse_judge_scores("not json at all") is None


class TestWeightedReward:
    def test_sums_with_weights(self):
        from rag_pipeline.mcts_reasoner import weighted_reward
        weights = {"precedent_alignment": 0.4, "factor_completeness": 0.3,
                   "logical_consistency": 0.2, "evidence_strength": 0.1}
        scores = {"precedent_alignment": 1.0, "factor_completeness": 0.5,
                  "logical_consistency": 0.0, "evidence_strength": 0.0}
        assert abs(weighted_reward(scores, weights) - 0.55) < 1e-9

    def test_missing_dims_default(self):
        from rag_pipeline.mcts_reasoner import weighted_reward
        weights = {"precedent_alignment": 0.4, "factor_completeness": 0.3,
                   "logical_consistency": 0.2, "evidence_strength": 0.1}
        assert abs(weighted_reward({}, weights) - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# ReasoningState tests
# ---------------------------------------------------------------------------

class TestReasoningState:
    def test_ucb1_explores_unvisited(self):
        from rag_pipeline.mcts_reasoner import ReasoningState
        s = ReasoningState(id="1", question="q")
        assert s.ucb1(parent_visits=100) == float("inf")

    def test_ucb1_balances_exploitation(self):
        from rag_pipeline.mcts_reasoner import ReasoningState
        s = ReasoningState(id="1", question="q", visits=10, total_reward=7.0)
        assert s.ucb1(parent_visits=100, exploration_constant=1.414) > s.average_reward
        assert s.average_reward == 0.7

    def test_is_leaf_without_children(self):
        from rag_pipeline.mcts_reasoner import ReasoningState
        assert ReasoningState(id="1", question="q").is_leaf


# ---------------------------------------------------------------------------
# MCTSReasoner integration tests (fakes, no network)
# ---------------------------------------------------------------------------

class FakeJudge:
    """Fake chat with .generate returning canned responses by prompt type."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        if "JSON array" in prompt:
            return '["issue one", "issue two", "issue three"]'
        if "Score 4 dimensions" in prompt:
            return ('{"precedent_alignment": 0.8, "factor_completeness": 0.7, '
                    '"logical_consistency": 0.9, "evidence_strength": 0.6}')
        return "FINAL ANSWER from reasoning path."


class FakeRag:
    """Fake LegalRAGQuery with .query returning static sources."""

    def query(self, question, top_k=3, include_analysis=False, **kwargs):
        return SimpleNamespace(
            sources=[{"excerpt": "Relevant legal precedent about employee status."},
                      {"excerpt": "Control test factors from case law."}],
        )


class TestMCTSReasoner:
    def test_query_include_analysis_false_no_unbound_error(self):
        """Regression: query(include_analysis=False) used to crash with
        UnboundLocalError on verification_degraded (scoped inside the
        include_analysis branch but referenced unconditionally)."""
        from rag_pipeline.rag_query import LegalRAGQuery, RAGResponse

        rag = object.__new__(LegalRAGQuery)
        rag.cache = None
        rag.metrics_collector = None
        rag.verifier = None
        rag.chat = None
        rag.confidence_gate = None
        rag.embeddings = SimpleNamespace(
            embed_text=lambda q: SimpleNamespace(embedding=[0.1, 0.2]),
        )
        rag.vector_store = SimpleNamespace(search=lambda **kw: [])
        rag.hybrid_retriever = SimpleNamespace(
            retrieve=lambda **kw: [
                SimpleNamespace(
                    id="r1", score=0.9, content="Relevant precedent",
                    metadata={"title": "Case A", "citation": "2001 SCC 59"},
                )
            ],
        )

        response = rag.query(
            question="Test legal question about employee status?",
            include_analysis=False,
        )
        assert isinstance(response, RAGResponse)
        assert response.status == "ok"

    def test_reason_completes_with_best_path(self):
        from rag_pipeline.mcts_reasoner import MCTSReasoner
        reasoner = MCTSReasoner(
            rag_query=FakeRag(),
            judge=FakeJudge(),
            n_simulations=5,
            max_depth=3,
        )
        result = reasoner.reason("Is this worker an employee or contractor?")

        assert result.status == "ok"
        assert result.answer  # synthesis produced an answer
        assert len(result.best_path) >= 1
        assert result.tree_statistics["total_simulations"] >= 1

    def test_best_path_dicts_carry_evidence(self):
        """Regression: to_dict() used to omit evidence, so query_reasoned()
        always produced empty sources for MCTS answers."""
        from rag_pipeline.mcts_reasoner import MCTSReasoner
        reasoner = MCTSReasoner(
            rag_query=FakeRag(),
            judge=FakeJudge(),
            n_simulations=5,
            max_depth=3,
        )
        result = reasoner.reason("Is this worker an employee or contractor?")

        assert len(result.best_path) >= 2
        non_root = result.best_path[1:]
        assert any(s.get("evidence") for s in non_root), (
            "best_path dicts must carry evidence so query_reasoned can "
            "emit sources (to_dict regression)"
        )

    def test_query_reasoned_emits_sources(self):
        """End-to-end: query_reasoned() must emit non-empty sources once
        best_path dicts carry evidence (to_dict regression)."""
        from rag_pipeline.rag_query import LegalRAGQuery, RAGResponse
        from types import SimpleNamespace

        class FakeQuery:
            def query(self, question, top_k=3, include_analysis=False, **kwargs):
                return SimpleNamespace(
                    sources=[
                        {"excerpt": "Relevant legal precedent about employee status."},
                        {"excerpt": "Control test factors from case law."},
                    ],
                )

            def _estimate_complexity(self, question):
                return "complex"

        rag = object.__new__(LegalRAGQuery)
        rag.chat = FakeJudge()
        rag.verifier = None
        rag.query = FakeQuery().query
        rag._estimate_complexity = FakeQuery()._estimate_complexity

        response = rag.query_reasoned(
            "Is this worker an employee or independent contractor?",
            n_simulations=5,
        )
        assert isinstance(response, RAGResponse)
        assert response.retrieval_mode == "mcts"
        assert response.sources, (
            "query_reasoned returned no sources — evidence is missing from "
            "best_path (to_dict regression)"
        )
        for s in response.sources:
            assert s.get("retrieved_by") == ["mcts"]

    def test_reason_builds_tree(self):
        from rag_pipeline.mcts_reasoner import MCTSReasoner
        reasoner = MCTSReasoner(
            rag_query=FakeRag(),
            judge=FakeJudge(),
            n_simulations=3,
            max_depth=3,
        )
        result = reasoner.reason("Test question")
        assert result.tree_statistics["total_states"] >= 4  # root + 3 issues min

    def test_reason_survives_judge_failure(self):
        from rag_pipeline.mcts_reasoner import MCTSReasoner
        from rag_pipeline.mcts_reasoner import ReasoningResult

        class ExplodingJudge:
            def generate(self, prompt, **kwargs):
                raise RuntimeError("judge down")

        reasoner = MCTSReasoner(
            rag_query=FakeRag(),
            judge=ExplodingJudge(),
            n_simulations=3,
            max_depth=3,
        )
        # Should not hang or crash; decomposition falls back to [question]
        result = reasoner.reason("Test question")
        assert isinstance(result, ReasoningResult)

    def test_simulate_cheap_proxy_skips_judge(self):
        from rag_pipeline.mcts_reasoner import MCTSReasoner

        class StrongRag:
            def query(self, question, **kwargs):
                return SimpleNamespace(sources=[
                    {"excerpt": "test question employee contractor control complete"}
                ] * 3)

        judge = FakeJudge()
        reasoner = MCTSReasoner(
            rag_query=StrongRag(),
            judge=judge,
            n_simulations=2,
            max_depth=2,
            judge_threshold=0.35,
        )
        reasoner.reason("test question")
        judge_prompts = [c for c in judge.calls if "Score 4 dimensions" in c]
        assert len(judge_prompts) == 0  # all nodes scored by cheap proxy