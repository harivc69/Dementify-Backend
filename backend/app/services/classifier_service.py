import sys
import os
from pathlib import Path
from typing import Dict, List, Union, Optional
import pandas as pd
import json

# Import the classifier from the local libs package
from app.libs.hierarchical_model_package.hierarchical_classifier import HierarchicalCognitiveClassifier

class ClassifierService:
    """
    Singleton service for the Hierarchical Cognitive Classifier.
    """
    
    _instance: Optional['ClassifierService'] = None
    _classifier: Optional[HierarchicalCognitiveClassifier] = None
    _is_initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._is_initialized:
            self._initialize()
            ClassifierService._is_initialized = True
    
    def _initialize(self):
        """Initialize the classifier (loads models)."""
        print("=" * 70)
        print("Initializing Hierarchical Cognitive Classifier...")
        print("=" * 70)
        try:
            self._classifier = HierarchicalCognitiveClassifier()
            print(f"Classifier loaded successfully on device: {self._classifier.device}")
        except Exception as e:
            print(f"Error loading classifier: {e}")
            raise
    
    @property
    def classifier(self) -> HierarchicalCognitiveClassifier:
        """Get the classifier instance."""
        if self._classifier is None:
            self._initialize()
        return self._classifier
    
    @property
    def is_ready(self) -> bool:
        """Check if the classifier is ready."""
        return self._classifier is not None
    
    def get_feature_list(self) -> List[str]:
        return self.classifier.get_feature_list()
    
    def get_feature_info(self) -> Dict:
        return self.classifier.get_feature_info()

    def get_model_info(self) -> Dict:
        return {
            "name": "Hierarchical Cognitive Classifier",
            "version": "Jan 5, 2026",
            "type": "3-stage Hierarchical"
        }
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """Run hierarchical prediction on a dataframe."""
        return self.classifier.predict(df, return_probabilities=True)

# Global singleton instance
_classifier_service: Optional[ClassifierService] = None

def get_classifier_service() -> ClassifierService:
    """Get or create the classifier service singleton."""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = ClassifierService()
    return _classifier_service

def initialize_classifier():
    """Pre-initialize the classifier (call at app startup)."""
    return get_classifier_service()
