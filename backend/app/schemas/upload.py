from pydantic import BaseModel

class UploadResponse(BaseModel):
    """Response model for file uploads."""
    message: str
    file_url: str
    filename: str
