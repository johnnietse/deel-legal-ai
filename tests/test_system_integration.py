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

# Add parent directory to path
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
        
        # Should be able to load config-defined data
        df = model.load_data()
        assert not df.empty
        
        # If trained model exists, verify inference
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

    def test_rag_pipeline_initialization(self):
        """Verify RAG pipeline initializes and connects to Pinecone"""
        try:
            pipeline = LegalRAGPipeline()
            assert pipeline.pinecone is not None
            # Check stats
            stats = pipeline.pinecone.get_stats()
            count = stats.total_vector_count
            print(f"\nPinecone Index Count: {count}")
            assert count > 0, "Pinecone index is empty"
            
        except Exception as e:
            pytest.fail(f"RAG Pipeline initialization failed: {e}")

if __name__ == "__main__":
    # Manually run if executed as script
    t = TestSystemIntegration()
    t.test_data_availability()
    print("✅ Data Verified")
    t.test_ml_pipeline_integration()
    print("✅ ML Pipeline Verified")
    t.test_rag_pipeline_initialization()
    print("✅ RAG Pipeline Verified")
    print("\nSYSTEM INTEGRATION TEST PASSED")
