"""
Custom Exception Classes for the Application

All exceptions include:
- message: User-friendly error message
- code: Error code for developers (from ErrorCode enum)
- status_code: HTTP status code
- detail: Additional technical details
"""

from fastapi import status
from typing import Optional
from app.core.error_codes import ErrorCode, get_error_message


class AppError(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self, 
        message: str = None,
        code: ErrorCode = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, 
        detail: str = None
    ):
        self.code = code.value if code else None
        self.message = message or (get_error_message(code) if code else "An unexpected error occurred")
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class ValidationError(AppError):
    """Exception raised when data validation fails."""
    
    def __init__(
        self, 
        message: str = None,
        code: ErrorCode = ErrorCode.VALIDATION_INVALID_DATA,
        detail: str = None
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=detail
        )


class AuthError(AppError):
    """Exception raised for authentication or authorization failures."""
    
    def __init__(
        self, 
        message: str = None,
        code: ErrorCode = ErrorCode.AUTH_UNAUTHORIZED,
        detail: str = None
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=detail
        )


class NotFoundError(AppError):
    """Exception raised when a resource is not found."""
    
    def __init__(
        self, 
        message: str = None,
        code: ErrorCode = ErrorCode.DB_NOT_FOUND,
        detail: str = None
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=detail
        )


class DatabaseError(AppError):
    """Exception raised for database-related errors."""
    
    def __init__(
        self, 
        message: str = None,
        code: ErrorCode = ErrorCode.DB_QUERY_ERROR,
        detail: str = None
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=detail
        )


class InferenceError(AppError):
    """Exception raised for model inference errors."""
    
    def __init__(
        self, 
        message: str = None,
        code: ErrorCode = ErrorCode.PREDICTION_FAILED,
        detail: str = None
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=detail
        )


class FileError(AppError):
    """Exception raised for file handling errors."""
    
    def __init__(
        self, 
        message: str = None,
        code: ErrorCode = ErrorCode.FILE_PARSE_ERROR,
        detail: str = None
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=detail
        )

