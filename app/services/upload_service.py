import os
import io
import json
import pandas as pd
from datetime import datetime
from fastapi import UploadFile
from app.core.config import settings
from app.core.exceptions import AppError, ValidationError
from app.services.data_service import DataService

class UploadService:

    def __init__(self, db):
        self.db = db
        self.collection = db.uploads
        # settings.UPLOAD_DIR is now absolute, so os.makedirs works correctly
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    async def save_upload(self, file: UploadFile, username: str) -> dict:
        """
        Save an uploaded file and record metadata.
        Automatically converts CSV and JSON files to model-ready format.
        """
        # Clean filename and add timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        original_filename = file.filename
        safe_filename = original_filename.replace(' ', '_')
        
        # Determine strict extension based on processing result (we always save as CSV if processed)
        is_processed = False
        final_extension = os.path.splitext(safe_filename)[1]
        
        try:
            content = await file.read()
            
            # Reset file cursor just in case, though we have the content bytes now
            await file.seek(0)

            df = None
            if safe_filename.lower().endswith('.csv'):
                # Process CSV
                df = DataService.parse_csv(content)
                is_processed = True
            elif safe_filename.lower().endswith('.json'):
                # Process JSON
                df = DataService.parse_json(content)
                is_processed = True
                # Change extension to .csv for saved file as we normalize to CSV
                base_name = os.path.splitext(safe_filename)[0]
                final_extension = ".csv"
                safe_filename = f"{base_name}.csv"

            if is_processed and df is not None:
                # Convert normalized DF back to CSV bytes
                # index=False to match model expectation (usually no index col unless specified)
                output_buffer = io.StringIO()
                df.to_csv(output_buffer, index=False)
                file_data = output_buffer.getvalue().encode('utf-8')
            else:
                # Use original content for non-supported files (or if we skipped processing)
                file_data = content

        except ValidationError as ve:
             # Propagate validation errors (invalid CSV/JSON structure or missing mandatory fields)
             raise ve
        except Exception as e:
            raise AppError(message="Could not process file", detail=str(e))

        # Final filename with timestamp
        filename = f"{timestamp}_{safe_filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, filename)

        try:
            with open(file_path, "wb") as buffer:
                buffer.write(file_data)
        except Exception as e:
            raise AppError(message="Could not save file to disk", detail=str(e))


        file_url = f"/uploads/{filename}" # Construct URL relative to API serving or static mount? 
        # Note: Original code had f"/{settings.UPLOAD_DIR}/{filename}". 
        # If UPLOAD_DIR is absolute, this url is wrong. 
        # Assuming frontend expects a relative URL path like /uploads/filename.csv
        # Let's hardcode the url prefix to '/uploads/' to be safe for serving.
        
        upload_doc = {
            "username": username,
            "filename": filename,
            "original_filename": original_filename,
            "file_url": file_url,
            "uploaded_at": datetime.utcnow(),
            "processed": is_processed
        }
        
        await self.collection.insert_one(upload_doc)
        return upload_doc
