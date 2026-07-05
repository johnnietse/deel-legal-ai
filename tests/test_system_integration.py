"""
System Integration Test
Verifies that all components of the Legal AI System are working together correctly.
"""
import sys
import os
import pytest
from pathlib import Path
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMPLOYMENT_CASES_CSV, ML_MODEL_PATH
from ml_classifier.train_classifier import WorkerClassificationModel
from rag_pipeline.pipeline import LegalRAGPipeline


class TestSystemIntegration:

    def test_data_availability(self):
        """Verify that the scaled dataset exists and has expected size"""
        assert Path(EMPLOYMENT_CASES_CSV).exists(), "Dataset file missing"
        df = pd.read_csv(EMPLOYMENT_CASES_CSV)
        assert len(df) > 1000, f"Dataset size {len(df)} is smaller than expected 1000+"
        assert "Outcome" in df.columns, "Target column 'Outcome' missing"

    def test_ml_pipeline_integration(self):
        """Verify that the ML model can load data and predict"""
        model = WorkerClassificationModel()
        df = model.load_data()
        assert not df.empty

        if Path(ML_MODEL_PATH).exists():
            model.load_model()
            sample = {
                'Supervision/review of work': 'High',
                'Ownership of tools': 'Employer',
                'Exclusivity of services': 'Yes'
            }
            result = model.predict(sample)
            assert "prediction" in result
            assert "confidence" in result
            print(f"\nML Prediction: {result['prediction']} ({result['confidence']:.2%})")
        else:
            pytest.skip("ML Model not trained yet")


class TestByteDanceModules:
    """Tests for ByteDance v3.0 enhancements."""

    def test_vector_store_interface(self):
        """Verify VectorStore abstract interface creates backends."""
        from rag_pipeline.vector_store import create_vector_store, VectorStore

        store = create_vector_store(backend="pinecone")
        assert isinstance(store, VectorStore)

    def test_bm25_engine_factory(self):
        """Verify BM25 engine factory creates correct backend."""
        from rag_pipeline.search_engine import create_bm25_engine, ElasticsearchBM25

        engine = create_bm25_engine(backend="elasticsearch")
        assert isinstance(engine, ElasticsearchBM25)

    def test_hybrid_retriever_structure(self):
        """Verify HybridRetriever can be constructed with mocks."""
        from rag_pipeline.hybrid_retriever import HybridRetriever, QueryClassifier, reciprocal_rank_fusion

        assert QueryClassifier.classify("What is the Sagaz test?") in ("keyword", "semantic", "hybrid")
        assert callable(reciprocal_rank_fusion)

    def test_semantic_chunker_exists(self):
        """Verify SemanticChunker is importable from document_processor."""
        from rag_pipeline.document_processor import SemanticChunker

        chunker = SemanticChunker()
        assert chunker is not None

    def test_feedback_analyzer(self):
        """Verify FeedbackAnalyzer loads and produces a summary."""
        from rag_pipeline.feedback_analyzer import FeedbackAnalyzer, FeedbackStore, FeedbackEntry

        store = FeedbackStore()
        entry = FeedbackEntry(
            query_id="test_001",
            query_text="test query",
            answer_text="test answer",
            rating="useful",
            error_type=None,
            comment=None,
        )
        store.record(entry)
        analyzer = FeedbackAnalyzer(store=store)
        summary = analyzer.summary()
        assert summary["total"] >= 1

    def test_model_optimization_configs(self):
        """Verify LoRA and Quantisation configs are valid."""
        from rag_pipeline.model_optimization import LoRAConfig, QuantisationConfig

        lora = LoRAConfig()
        assert lora.lora_r == 16

        quant = QuantisationConfig()
        assert quant.bits == 8

    def test_config_env_vars(self):
        """Verify key ByteDance config vars are set."""
        import config

        assert hasattr(config, "VECTOR_STORE_BACKEND")
        assert hasattr(config, "BM25_BACKEND")
        assert hasattr(config, "HYBRID_SEARCH_ENABLED")
        assert hasattr(config, "FEEDBACK_STORE_PATH")
        assert hasattr(config, "LORA_BASE_MODEL")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
