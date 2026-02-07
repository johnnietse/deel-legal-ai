# Tests for Deel Lab Legal AI System
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentProcessor:
    """Tests for the document processor"""
    
    def test_chunking_basic(self):
        """Test basic text chunking"""
        from rag_pipeline.document_processor import LegalDocumentProcessor
        
        processor = LegalDocumentProcessor(chunk_size=100, chunk_overlap=10)
        
        sample_text = """
        This is a sample legal document.
        
        It contains multiple paragraphs that should be chunked appropriately.
        
        Each chunk should maintain semantic coherence while respecting token limits.
        """
        
        chunks = processor.chunk_text(sample_text, "test_doc")
        
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
    
    def test_legal_section_detection(self):
        """Test detection of legal sections"""
        from rag_pipeline.document_processor import LegalDocumentProcessor
        
        processor = LegalDocumentProcessor()
        
        # Test various section headers
        assert processor._detect_section("FACTS: The plaintiff claims...") == "facts"
        assert processor._detect_section("ANALYSIS of the evidence...") == "analysis"
        assert processor._detect_section("CONCLUSION: Based on...") == "conclusion"


class TestClassificationModel:
    """Tests for the worker classification model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        from ml_classifier.train_classifier import WorkerClassificationModel
        
        model = WorkerClassificationModel()
        assert model is not None
        assert not model.is_trained
    
    def test_feature_columns(self):
        """Test feature columns are defined"""
        from ml_classifier.train_classifier import WorkerClassificationModel
        
        model = WorkerClassificationModel()
        assert len(model.FEATURE_COLUMNS) == 10
        assert "Supervision/review of work" in model.FEATURE_COLUMNS


class TestAPI:
    """Tests for the FastAPI service"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_classification_factors(self, client):
        """Test classification factors endpoint"""
        response = client.get("/classify/factors")
        assert response.status_code == 200


class TestEmbeddings:
    """Tests for Gemini embeddings (requires API key)"""
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath(".env").exists(),
        reason="No .env file with API keys"
    )
    def test_embedding_generation(self):
        """Test embedding generation"""
        from rag_pipeline.embeddings import GeminiEmbeddings
        
        embedder = GeminiEmbeddings()
        result = embedder.embed_text("Employment law in Ontario")
        
        assert result.embedding is not None
        assert len(result.embedding) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
