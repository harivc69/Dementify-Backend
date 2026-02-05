import os
from datetime import datetime
from fastapi import UploadFile
from app.core.config import settings

class UploadService:
    def __init__(self, db):
        self.db = db
        self.collection = db.uploads
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    async def save_upload(self, file: UploadFile, username: str) -> dict:
        """Save an uploaded file and record metadata."""
        # Clean filename and add timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        safe_filename = file.filename.replace(' ', '_')
        filename = f"{timestamp}_{safe_filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)

        try:
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
        except Exception as e:
            raise Exception(f"Could not save file: {str(e)}")

        file_url = f"/{settings.UPLOAD_DIR}/{filename}"
        
        upload_doc = {
            "username": username,
            "filename": filename,
            "file_url": file_url,
            "uploaded_at": datetime.utcnow()
        }
        
        await self.collection.insert_one(upload_doc)
        return upload_doc
