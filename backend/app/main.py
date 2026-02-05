import os

# Suppress OpenMP duplicate library error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.db.database import db_manager
from app.routes.api import api_router
from app.services.classifier_service import initialize_classifier

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade Hierarchical Cognitive Impairment Classification API",
    version="1.0.0"
)

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
