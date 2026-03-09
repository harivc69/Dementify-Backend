from fastapi import APIRouter
from app.routes import auth, predict, shap, upload

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(predict.router, prefix="/inference/predict", tags=["Prediction"])
api_router.include_router(shap.router, prefix="/inference/predict/shap", tags=["SHAP Explanations"])
api_router.include_router(upload.router, prefix="/files", tags=["Files"])
