from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class UploadDB(BaseModel):
    """Internal database representation of a File Upload."""
    username: str
    filename: str
    file_url: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    
    class Config:
        populate_by_name = True
