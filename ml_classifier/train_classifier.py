# ML Classifier - Random Forest for Worker Classification
"""
Random Forest classifier for predicting worker classification 
(employee vs. independent contractor) in employment law cases.

This implements:
"Developed and fine-tuned a Random Forest classifier using scikit-learn 
and pandas on a tabular dataset of 700+ annotated employment law cases 
to predict work classification (employee vs. independent contractor), 
achieving a preliminary accuracy of 88%"
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMPLOYMENT_CASES_CSV, ML_MODEL_PATH, ML_FEATURE_IMPORTANCE_PATH,
    ML_TEST_SIZE, ML_RANDOM_STATE, MODELS_DIR, LOG_FORMAT, LOG_LEVEL
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore')


class WorkerClassificationModel:
    """
    Random Forest classifier for worker classification.
    
    Features used for classification (based on Sagaz test and common factors):
    - Supervision/review of work
    - Ability to hire employees
    - Delegation of tasks
    - Ownership of tools
    - Chance of profit
    - Risk of loss
    - Exclusivity of services
    - Who sets the work hours
    - Where the work is performed
    - Is the worker required to wear a uniform?
    """
    
    # Feature columns from the employment cases dataset
    FEATURE_COLUMNS = [
        'Supervision/review of work',
        'Ability to hire employees',
        'Delegation of tasks',
        'Ownership of tools',
        'Chance of profit',
        'Risk of loss',
        'Exclusivity of services',
        'Who sets the work hours',
        'Where the work is performed',
        'Is the worker required to wear a uniform?'
    ]
    
    TARGET_COLUMN = 'Outcome'
    
    # Legal factor descriptions for interpretability
    FACTOR_DESCRIPTIONS = {
        'Supervision/review of work': 'Degree of control/supervision by the employer',
        'Ability to hire employees': 'Worker can hire helpers or subcontract',
        'Delegation of tasks': 'Worker can delegate tasks to others',
        'Ownership of tools': 'Who provides tools/equipment',
        'Chance of profit': 'Worker has opportunity for profit beyond wages',
        'Risk of loss': 'Worker bears financial risk of loss',
        'Exclusivity of services': 'Worker provides services to one entity only',
        'Who sets the work hours': 'Control over work schedule',
        'Where the work is performed': 'Control over work location',
        'Is the worker required to wear a uniform?': 'Appearance/uniform requirements'
    }
    
    def __init__(self):
        self.model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.target_encoder = LabelEncoder()
        self.feature_importance: Dict[str, float] = {}
        self.training_stats: Dict[str, Any] = {}
        self.is_trained = False
    
    def load_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and validate the employment cases dataset.
        
        Args:
            csv_path: Path to CSV file (uses default if None)
            
        Returns:
            DataFrame with employment cases
        """
        csv_path = Path(csv_path) if csv_path else EMPLOYMENT_CASES_CSV
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} cases")
        
        # Validate required columns
        missing_cols = set(self.FEATURE_COLUMNS + [self.TARGET_COLUMN]) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset for training.
        
        - Handles missing values
        - Encodes categorical variables
        - Returns feature matrix and target vector
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (X, y) numpy arrays
        """
        # Work on a copy
        data = df.copy()
        
        # Select available feature columns
        available_features = [col for col in self.FEATURE_COLUMNS if col in data.columns]
        logger.info(f"Using {len(available_features)} features")
        
        if not available_features:
            raise ValueError("No feature columns found in dataset")
        
        # Filter rows with target values
        data = data[data[self.TARGET_COLUMN].notna()]
        
        # Handle missing values in features
        for col in available_features:
            if data[col].isna().any():
                # Fill with mode for categorical data
                mode_value = data[col].mode()
                if len(mode_value) > 0:
                    data[col] = data[col].fillna(mode_value[0])
                else:
                    data[col] = data[col].fillna('Unknown')
        
        # Encode features
        X_encoded = []
        for col in available_features:
            if data[col].dtype == 'object':
                # Categorical encoding
                le = LabelEncoder()
                encoded = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
                X_encoded.append(encoded)
            else:
                # Numeric - just append
                X_encoded.append(data[col].values)
        
        X = np.column_stack(X_encoded).astype(np.float64) if X_encoded else np.array([])
        
        # Encode target
        y = self.target_encoder.fit_transform(data[self.TARGET_COLUMN].astype(str))
        
        self.training_stats["feature_columns"] = available_features
        self.training_stats["n_samples"] = len(y)
        self.training_stats["target_classes"] = list(self.target_encoder.classes_)
        
        logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target classes: {self.target_encoder.classes_}")
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hyperparameter_tuning: bool = True
    ) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Feature matrix
            y: Target vector
            hyperparameter_tuning: Whether to perform grid search
            
        Returns:
            Training results with metrics
        """
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        self.training_stats["train_size"] = len(y_train)
        self.training_stats["test_size"] = len(y_test)
        
        if hyperparameter_tuning:
            # Grid search for best parameters
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            logger.info("Performing hyperparameter tuning...")
            
            rf = RandomForestClassifier(random_state=ML_RANDOM_STATE)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.training_stats["best_params"] = grid_search.best_params_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Default model
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=ML_RANDOM_STATE
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "classification_report": classification_report(
                y_test, y_pred, 
                target_names=self.target_encoder.classes_,
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.training_stats["metrics"] = results
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        results["cv_accuracy_mean"] = cv_scores.mean()
        results["cv_accuracy_std"] = cv_scores.std()
        
        logger.info(f"Test Accuracy: {results['accuracy']:.3f}")
        logger.info(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} (+/- {results['cv_accuracy_std']:.3f})")
        
        # Feature importance
        self._calculate_feature_importance()
        
        self.is_trained = True
        return results
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance"""
        if self.model is None:
            return
        
        feature_cols = self.training_stats.get("feature_columns", [])
        importances = self.model.feature_importances_
        
        self.feature_importance = {
            col: float(imp) 
            for col, imp in zip(feature_cols, importances)
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info("Feature Importance (top 5):")
        for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:5]):
            logger.info(f"  {i+1}. {feature}: {importance:.3f}")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict worker classification for a single case.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Prediction result with probability and factors
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Prepare feature vector
        feature_cols = self.training_stats["feature_columns"]
        X = []
        
        for col in feature_cols:
            value = features.get(col, 'Unknown')
            
            if col in self.label_encoders:
                try:
                    encoded = self.label_encoders[col].transform([str(value)])[0]
                except ValueError:
                    # Unknown category
                    encoded = 0
                X.append(encoded)
            else:
                X.append(value if isinstance(value, (int, float)) else 0)
        
        X = np.array(X).reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        predicted_class = self.target_encoder.inverse_transform([prediction])[0]
        
        # Get probability for each class
        class_probs = {
            cls: float(prob)
            for cls, prob in zip(self.target_encoder.classes_, probabilities)
        }
        
        # Get top contributing factors
        top_factors = self._get_top_contributing_factors(features)
        
        return {
            "prediction": predicted_class,
            "confidence": float(max(probabilities)),
            "class_probabilities": class_probs,
            "top_factors": top_factors
        }
    
    def _get_top_contributing_factors(
        self, 
        features: Dict[str, Any], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get the top contributing factors for a prediction"""
        factors = []
        
        for feature, importance in list(self.feature_importance.items())[:top_k]:
            factors.append({
                "feature": feature,
                "importance": importance,
                "value": features.get(feature, 'Unknown'),
                "description": self.FACTOR_DESCRIPTIONS.get(feature, "")
            })
        
        return factors
    
    def save_model(self, model_path: Optional[str] = None):
        """Save trained model to disk"""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        model_path = Path(model_path) if model_path else ML_MODEL_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "target_encoder": self.target_encoder,
            "feature_importance": self.feature_importance,
            "training_stats": self.training_stats
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance as JSON
        importance_path = Path(model_path).parent / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump({
                "feature_importance": self.feature_importance,
                "factor_descriptions": self.FACTOR_DESCRIPTIONS,
                "training_stats": {
                    k: v for k, v in self.training_stats.items()
                    if k != "metrics"  # Metrics can be large
                }
            }, f, indent=2)
        logger.info(f"Feature importance saved to {importance_path}")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load trained model from disk"""
        model_path = Path(model_path) if model_path else ML_MODEL_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data["model"]
        self.label_encoders = model_data["label_encoders"]
        self.target_encoder = model_data["target_encoder"]
        self.feature_importance = model_data["feature_importance"]
        self.training_stats = model_data["training_stats"]
        self.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the trained model"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "accuracy": self.training_stats.get("metrics", {}).get("accuracy", 0),
            "n_samples": self.training_stats.get("n_samples", 0),
            "n_features": len(self.training_stats.get("feature_columns", [])),
            "target_classes": self.training_stats.get("target_classes", []),
            "top_features": list(self.feature_importance.items())[:5]
        }


def train_and_evaluate():
    """Train and evaluate the worker classification model"""
    print("\n" + "="*60)
    print("WORKER CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    model = WorkerClassificationModel()
    
    try:
        # Load data
        df = model.load_data()
        print(f"\n📊 Dataset loaded: {len(df)} cases")
        
        # Preprocess
        X, y = model.preprocess_data(df)
        print(f"📐 Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # Train with hyperparameter tuning
        results = model.train(X, y, hyperparameter_tuning=True)
        
        print("\n" + "-"*40)
        print("TRAINING RESULTS")
        print("-"*40)
        print(f"✅ Accuracy: {results['accuracy']:.2%}")
        print(f"📈 Precision: {results['precision']:.2%}")
        print(f"📊 Recall: {results['recall']:.2%}")
        print(f"📉 F1 Score: {results['f1_score']:.2%}")
        print(f"🔄 CV Accuracy: {results['cv_accuracy_mean']:.2%} (+/- {results['cv_accuracy_std']:.2%})")
        
        print("\n" + "-"*40)
        print("FEATURE IMPORTANCE (Legal Factors)")
        print("-"*40)
        for i, (feature, importance) in enumerate(model.feature_importance.items()):
            bar = "█" * int(importance * 40)
            print(f"{i+1:2}. {feature[:35]:35} {importance:.3f} {bar}")
        
        # Save model
        model.save_model()
        print(f"\n💾 Model saved to {ML_MODEL_PATH}")
        
        return model, results
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the employment_cases.csv file exists")
        return None, None


def demo_prediction():
    """Demo prediction with sample case"""
    model = WorkerClassificationModel()
    
    try:
        model.load_model()
        
        # Sample case for prediction
        sample_case = {
            'Supervision/review of work': 'High',
            'Ability to hire employees': 'No',
            'Delegation of tasks': 'No',
            'Ownership of tools': 'Employer',
            'Chance of profit': 'No',
            'Risk of loss': 'No',
            'Exclusivity of services': 'Yes',
            'Who sets the work hours': 'Employer',
            'Where the work is performed': 'Employer premises',
            'Is the worker required to wear a uniform?': 'Yes'
        }
        
        result = model.predict(sample_case)
        
        print("\n" + "="*60)
        print("WORKER CLASSIFICATION PREDICTION")
        print("="*60)
        
        print("\n📋 Case Factors:")
        for factor, value in sample_case.items():
            print(f"   • {factor}: {value}")
        
        print(f"\n🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        
        print("\n📈 Class Probabilities:")
        for cls, prob in result['class_probabilities'].items():
            print(f"   • {cls}: {prob:.2%}")
        
        print("\n🔍 Key Contributing Factors:")
        for factor in result['top_factors']:
            print(f"   • {factor['feature']}: {factor['value']}")
            print(f"     Importance: {factor['importance']:.3f}")
        
    except FileNotFoundError:
        print("Model not found. Run training first.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Worker Classification Model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--demo", action="store_true", help="Run demo prediction")
    
    args = parser.parse_args()
    
    if args.train or not (args.train or args.demo):
        train_and_evaluate()
    
    if args.demo:
        demo_prediction()
