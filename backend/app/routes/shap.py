from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from app.schemas.predict import PredictionRequest
from app.schemas.shap import ShapResponse
from app.routes.deps import get_current_user, get_classifier
from app.services.shap_service import ShapService
from app.services.data_service import DataService
from datetime import datetime
import json
import traceback

router = APIRouter()

def get_shap_service(classifier = Depends(get_classifier)):
    return ShapService(classifier)

@router.post("", response_model=ShapResponse)
async def predict_shap_unified(
    request: Request,
    username: str = Depends(get_current_user),
    shap_service: ShapService = Depends(get_shap_service)
):
    try:
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            try:
                body = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid JSON body")
                
            # Expecting {"data": ...} structure based on PredictionRequest schema
            if "data" not in body:
                 # If list passed directly, treat as data
                 if isinstance(body, list):
                     raw_data = body
                 else:
                     raise HTTPException(400, "JSON body must contain 'data' field or be a list of records")
            else:
                raw_data = body["data"]

            df = DataService.parse_json(json.dumps(raw_data).encode())
            
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            if not file:
                raise HTTPException(400, "Missing 'file' field in form data")
            
            content = await file.read()
            if file.filename.lower().endswith('.csv'):
                df = DataService.parse_csv(content)
            elif file.filename.lower().endswith('.json'):
                df = DataService.parse_json(content)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or JSON.")
        else:
            raise HTTPException(400, f"Unsupported Content-Type: {content_type}. Use 'application/json' or 'multipart/form-data'.")
            
        results = shap_service.get_explanations(df)
        return {
            "success": True,
            "result": results[0] if results else None,
            "results": results,
            "timestamp": datetime.utcnow(),
            "model_info": shap_service.classifier_service.get_model_info()
        }
    except HTTPException:
        # Re-raise HTTP exceptions (e.g., 400 errors)
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")
