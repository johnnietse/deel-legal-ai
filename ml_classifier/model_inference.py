# ML Classifier - Inference API
"""
Inference API for worker classification model.
Provides REST endpoints for predictions.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_classifier.train_classifier import WorkerClassificationModel
from config import ML_MODEL_PATH, LOG_FORMAT, LOG_LEVEL

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationRequest:
    """Request for worker classification"""
    supervision_review: str = "Unknown"
    ability_hire: str = "Unknown"
    delegation_tasks: str = "Unknown"
    ownership_tools: str = "Unknown"
    chance_profit: str = "Unknown"
    risk_loss: str = "Unknown"
    exclusivity_services: str = "Unknown"
    work_hours_setter: str = "Unknown"
    work_location: str = "Unknown"
    uniform_required: str = "Unknown"
    
    def to_model_features(self) -> Dict[str, str]:
        """Convert to model feature format"""
        return {
            'Supervision/review of work': self.supervision_review,
            'Ability to hire employees': self.ability_hire,
            'Delegation of tasks': self.delegation_tasks,
            'Ownership of tools': self.ownership_tools,
            'Chance of profit': self.chance_profit,
            'Risk of loss': self.risk_loss,
            'Exclusivity of services': self.exclusivity_services,
            'Who sets the work hours': self.work_hours_setter,
            'Where the work is performed': self.work_location,
            'Is the worker required to wear a uniform?': self.uniform_required
        }


@dataclass
class ClassificationResponse:
    """Response from worker classification"""
    prediction: str
    confidence: float
    is_employee: bool
    class_probabilities: Dict[str, float]
    contributing_factors: List[Dict[str, Any]]
    legal_interpretation: str
    

class WorkerClassificationAPI:
    """
    API for worker classification predictions.
    
    Provides:
    - Single prediction endpoint
    - Batch prediction
    - Model information
    - Legal interpretation
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = WorkerClassificationModel()
        self.model_path = Path(model_path) if model_path else ML_MODEL_PATH
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            self.model.load_model(str(self.model_path))
            logger.info("Classification model loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Model not found at {self.model_path}")
            logger.info("Model will need to be trained before predictions")
    
    def _generate_legal_interpretation(
        self, 
        prediction: str, 
        confidence: float,
        top_factors: List[Dict[str, Any]]
    ) -> str:
        """Generate legal interpretation of the classification"""
        factors_str = ", ".join([f["feature"] for f in top_factors[:3]])
        
        if "employee" in prediction.lower():
            interpretation = f"""Based on the provided factors, this worker is likely classified as an EMPLOYEE 
under Ontario employment law (confidence: {confidence:.0%}). 

The key factors supporting this classification are: {factors_str}.

Under the Sagaz test (671122 Ontario Ltd. v. Sagaz Industries Canada Inc., 2001 SCC 59), 
the central question is whether the worker is performing services as a person in business 
on their own account, or as part of the employer's business.

In this case, the level of control exercised by the putative employer and the worker's 
lack of independent economic activity suggest an employment relationship.

⚠️ DISCLAIMER: This is an automated assessment and should not be considered legal advice. 
Consult a qualified employment lawyer for definitive classification."""
        else:
            interpretation = f"""Based on the provided factors, this worker is likely classified as an INDEPENDENT CONTRACTOR 
under Ontario employment law (confidence: {confidence:.0%}).

The key factors supporting this classification are: {factors_str}.

Under the Sagaz test (671122 Ontario Ltd. v. Sagaz Industries Canada Inc., 2001 SCC 59), 
this worker appears to be performing services as a person in business on their own account.

Key indicators include the worker's control over their work, ownership of tools, 
and potential for profit/loss in their business operations.

⚠️ DISCLAIMER: This is an automated assessment and should not be considered legal advice. 
Consult a qualified employment lawyer for definitive classification."""
        
        return interpretation
    
    def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        """
        Classify a single worker case.
        
        Args:
            request: Classification request with case factors
            
        Returns:
            ClassificationResponse with prediction and analysis
        """
        if not self.model.is_trained:
            raise RuntimeError("Model not trained. Train the model first.")
        
        features = request.to_model_features()
        result = self.model.predict(features)
        
        prediction = result["prediction"]
        confidence = result["confidence"]
        is_employee = "employee" in prediction.lower()
        
        interpretation = self._generate_legal_interpretation(
            prediction, confidence, result["top_factors"]
        )
        
        return ClassificationResponse(
            prediction=prediction,
            confidence=confidence,
            is_employee=is_employee,
            class_probabilities=result["class_probabilities"],
            contributing_factors=result["top_factors"],
            legal_interpretation=interpretation
        )
    
    def classify_batch(
        self, 
        requests: List[ClassificationRequest]
    ) -> List[ClassificationResponse]:
        """Classify multiple cases"""
        return [self.classify(req) for req in requests]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model.is_trained:
            return {
                "status": "not_loaded",
                "message": "Model needs to be trained"
            }
        
        summary = self.model.get_model_summary()
        return {
            "status": "loaded",
            "accuracy": summary.get("accuracy"),
            "n_training_samples": summary.get("n_samples"),
            "n_features": summary.get("n_features"),
            "target_classes": summary.get("target_classes"),
            "top_features": [
                {"feature": name, "importance": imp}
                for name, imp in summary.get("top_features", [])
            ]
        }
    
    def get_feature_definitions(self) -> Dict[str, str]:
        """Get definitions of classification features"""
        return WorkerClassificationModel.FACTOR_DESCRIPTIONS


def create_sample_request() -> ClassificationRequest:
    """Create a sample request for testing"""
    return ClassificationRequest(
        supervision_review="High",
        ability_hire="No",
        delegation_tasks="No",
        ownership_tools="Employer",
        chance_profit="No",
        risk_loss="No",
        exclusivity_services="Yes",
        work_hours_setter="Employer",
        work_location="Employer premises",
        uniform_required="Yes"
    )


def main():
    """Test the inference API"""
    print("\n" + "="*60)
    print("WORKER CLASSIFICATION INFERENCE API")
    print("="*60)
    
    api = WorkerClassificationAPI()
    
    # Get model info
    info = api.get_model_info()
    print(f"\n📊 Model Status: {info['status']}")
    
    if info['status'] == 'loaded':
        print(f"   Accuracy: {info.get('accuracy', 0):.2%}")
        print(f"   Training Samples: {info.get('n_training_samples', 0)}")
        print(f"   Features: {info.get('n_features', 0)}")
        
        # Run sample prediction
        request = create_sample_request()
        print("\n📋 Sample Case:")
        features = request.to_model_features()
        for k, v in features.items():
            print(f"   • {k}: {v}")
        
        try:
            response = api.classify(request)
            
            print(f"\n🎯 Prediction: {response.prediction}")
            print(f"📊 Confidence: {response.confidence:.2%}")
            print(f"👤 Is Employee: {response.is_employee}")
            
            print("\n📈 Class Probabilities:")
            for cls, prob in response.class_probabilities.items():
                print(f"   • {cls}: {prob:.2%}")
            
            print("\n⚖️ Legal Interpretation:")
            print(response.legal_interpretation)
            
        except RuntimeError as e:
            print(f"\n❌ Error: {e}")
    else:
        print("\n⚠️ Model not trained. Run train_classifier.py first.")


if __name__ == "__main__":
    main()
