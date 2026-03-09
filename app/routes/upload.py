from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, status
from app.schemas.upload import UploadResponse
from app.routes.deps import get_current_user, get_database
from app.services.upload_service import UploadService
from app.core.exceptions import ValidationError, AppError


router = APIRouter()

@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    username: str = Depends(get_current_user),
    db = Depends(get_database)
):
    if not file.filename:
        raise ValidationError("No file selected")

    upload_service = UploadService(db)
    result = await upload_service.save_upload(file, username)
    return {
        "message": "File uploaded successfully ✅",
        "file_url": result["file_url"],
        "filename": result["filename"]
    }

