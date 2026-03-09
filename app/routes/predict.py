from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Request
import json
from app.schemas.predict import PredictionRequest, PredictionResponse, HealthResponse, FeaturesResponse
from app.routes.deps import get_current_user, get_classifier
from app.services.classifier_service import ClassifierService
from app.services.data_service import DataService
from app.core.exceptions import ValidationError, InferenceError
from app.core.error_codes import ErrorCode

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
    content_type = request.headers.get("content-type", "")
    
    try:
        if "application/json" in content_type:
            try:
                body = await request.json()
            except json.JSONDecodeError:
                raise ValidationError(
                    message="Invalid JSON body",
                    code=ErrorCode.VALIDATION_INVALID_JSON
                )
                
            # Expecting {"data": ...} structure based on PredictionRequest schema
            if "data" not in body:
                    # If list passed directly, treat as data
                    if isinstance(body, list):
                        raw_data = body
                    else:
                        raise ValidationError(
                            message="Invalid request format",
                            code=ErrorCode.VALIDATION_MISSING_FIELD,
                            detail="JSON body must contain 'data' field or be a list of records"
                        )
            else:
                raw_data = body["data"]
            
            df = DataService.parse_json(json.dumps(raw_data).encode())
            
        elif "multipart/form-data" in content_type:
            form = await request.form()
            file = form.get("file")
            if not file:
                raise ValidationError(
                    message="Missing file",
                    code=ErrorCode.VALIDATION_MISSING_FIELD,
                    detail="Please include a 'file' field in form data"
                )
            
            content = await file.read()
            if file.filename.lower().endswith('.csv'):
                df = DataService.parse_csv(content)
            elif file.filename.lower().endswith('.json'):
                df = DataService.parse_json(content)
            else:
                raise ValidationError(
                    message="Unsupported file format",
                    code=ErrorCode.FILE_UNSUPPORTED_TYPE,
                    detail="Please upload a CSV or JSON file"
                )
        else:
            raise ValidationError(
                message="Unsupported content type",
                code=ErrorCode.VALIDATION_INVALID_FORMAT,
                detail=f"Use 'application/json' or 'multipart/form-data' instead of '{content_type}'"
            )

        results = classifier.predict_from_dataframe(df)
        return {
            "success": True,
            "count": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    except ValidationError:
        raise
    except Exception as e:
        raise InferenceError(
            message="Prediction failed",
            code=ErrorCode.PREDICTION_FAILED,
            detail=str(e)
        )


