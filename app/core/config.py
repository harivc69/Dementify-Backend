import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project Settings
    PROJECT_NAME: str = "Dementify Backend"
    API_STR: str = "/api"
    
    # Database
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "Dementify"

    # JWT Security
    JWT_SECRET: str = "your-secret-key-change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Upload Settings
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
