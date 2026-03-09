import os

# Suppress OpenMP duplicate library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------
# PATCH: Fix for 'passlib' compatibility with 'bcrypt' > 4.0.0
# passlib 1.7.4 relies on bcrypt.__about__ which was removed.
# ---------------------------------------------------------
import bcrypt
try:
    bcrypt.__about__
except AttributeError:
    class About:
        __version__ = bcrypt.__version__
    bcrypt.__about__ = About()
# ---------------------------------------------------------

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.db.database import db_manager
from app.routes.api import api_router
from app.services.classifier_service import initialize_classifier
from app.core.exceptions import AppError



app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade Hierarchical Cognitive Impairment Classification API",
    version="1.0.0"
)

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code or f"HTTP_{exc.status_code}",
                "message": exc.message,
                "detail": exc.detail
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "detail": None
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": 500,
                "message": "An unexpected error occurred",
                "detail": str(exc)
            }
        }
    )



# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount(f"/{settings.UPLOAD_DIR}", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Lifecycle events
@app.on_event("startup")
async def startup_db_client():
    await db_manager.connect_to_mongo()
    # Initialize the classifier at startup
    print("🚀 Initializing cognitive classifier...")
    try:
        initialize_classifier()
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize classifier: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    await db_manager.close_mongo_connection()

# Include aggregated router
app.include_router(api_router, prefix=settings.API_STR)

@app.get("/")
async def root():
    return {
        "message": f"{settings.PROJECT_NAME} is running",
        "docs": "/docs",
        "api_v1": settings.API_STR
    }
