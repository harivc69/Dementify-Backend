from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Request
import json
from app.schemas.predict import PredictionRequest, PredictionResponse, HealthResponse, FeaturesResponse
from app.routes.deps import get_current_user, get_classifier
from app.services.classifier_service import ClassifierService
from app.services.data_service import DataService
from datetime import datetime

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(classifier: ClassifierService = Depends(get_classifier)):
    return {
        "status": "healthy" if classifier.is_ready else "unhealthy",
        "model_loaded": classifier.is_ready,
        "device": str(classifier.classifier.device) if classifier.is_ready else None
    }

@router.get("/features", response_model=FeaturesResponse)
async def get_features(classifier: ClassifierService = Depends(get_classifier)):
    features = classifier.get_feature_list()
    return {
        "total_features": len(features),
        "features": features
    }

@router.post("/data", response_model=PredictionResponse)
async def predict_unified(
    request: Request,
    username: str = Depends(get_current_user),
    classifier: ClassifierService = Depends(get_classifier)
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

        results = classifier.predict_from_dataframe(df)
        return {
            "success": True,
            "count": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
