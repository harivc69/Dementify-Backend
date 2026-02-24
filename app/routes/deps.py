from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from app.core.config import settings
from app.db.database import get_database
from app.services.classifier_service import get_classifier_service, ClassifierService

from app.core.exceptions import AuthError
from app.core.error_codes import ErrorCode

# We keep this for Swagger UI support, but our main logic will check cookies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

async def get_current_user(request: Request):
    """Validate JWT token from HttpOnly Cookie and return username."""
    token = request.cookies.get("access_token")
    
    if not token:
        raise AuthError(
            message="Not authenticated",
            code=ErrorCode.AUTH_TOKEN_INVALID
        )
        
    # Remove 'Bearer ' prefix if present (cookies usually just have the token, but we set it with Bearer in auth.py)
    if token.startswith("Bearer "):
        token = token.split(" ")[1]

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise AuthError(
                message="Invalid authentication token",
                code=ErrorCode.AUTH_TOKEN_INVALID
            )
        return username
    except JWTError:
        raise AuthError(
            message="Invalid authentication token",
            code=ErrorCode.AUTH_TOKEN_INVALID
        )


def get_classifier() -> ClassifierService:
    """Dependency to get the classifier service singleton."""
    return get_classifier_service()

