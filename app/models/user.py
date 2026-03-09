from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserDB(BaseModel):
    """Internal database representation of a User."""
    username: str
    email: EmailStr
    hashed_password: str
    full_name: Optional[str] = None
    role: str = "user"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
