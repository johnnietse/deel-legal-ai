# ML Classifier Package
"""
Worker Classification ML Module for Deel Lab

Provides Random Forest classification for determining whether
workers are employees or independent contractors.
"""

from ml_classifier.train_classifier import WorkerClassificationModel
from ml_classifier.model_inference import (
    WorkerClassificationAPI,
    ClassificationRequest,
    ClassificationResponse
)

__all__ = [
    "WorkerClassificationModel",
    "WorkerClassificationAPI",
    "ClassificationRequest",
    "ClassificationResponse"
]

__version__ = "1.0.0"
