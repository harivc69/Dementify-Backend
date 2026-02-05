from pydantic import BaseModel
from typing import Dict, List, Union, Optional
from datetime import datetime

class PatientData(BaseModel):
    """Single patient data model."""
    class Config:
        extra = "allow"

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: Union[Dict, List[Dict]]

class PredictionResponse(BaseModel):
    """Batch prediction response."""
    success: bool
    count: int
    results: List[Dict]
    timestamp: str
    model_info: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: Optional[str] = None

class FeaturesResponse(BaseModel):
    total_features: int
    features: List[str]
