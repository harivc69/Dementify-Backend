from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class ShapResponse(BaseModel):
    """Response model for SHAP explanations."""
    success: bool
    result: Optional[Dict] = None
    results: Optional[List[Dict]] = None
    timestamp: datetime
    model_info: Optional[Dict] = None
