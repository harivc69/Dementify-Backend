"""
Centralized Error Codes for the Application

Error codes follow this convention:
- VAL_1xxx: Validation errors
- FILE_2xxx: File handling errors
- API_3xxx: API/Inference errors
- AUTH_4xxx: Authentication errors
- DB_5xxx: Database errors

These codes are included in API responses for developer debugging
while user-friendly messages are shown in the UI.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes for the application."""
    
    # Validation Errors (1xxx)
    VALIDATION_MISSING_COLUMNS = "VAL_1001"
    VALIDATION_INVALID_FORMAT = "VAL_1002"
    VALIDATION_EMPTY_VALUES = "VAL_1003"
    VALIDATION_INVALID_JSON = "VAL_1004"
    VALIDATION_INVALID_DATA = "VAL_1005"
    VALIDATION_MISSING_FIELD = "VAL_1006"
    
    # File Errors (2xxx)
    FILE_UNSUPPORTED_TYPE = "FILE_2001"
    FILE_PARSE_ERROR = "FILE_2002"
    FILE_UPLOAD_FAILED = "FILE_2003"
    FILE_NOT_FOUND = "FILE_2004"
    FILE_TOO_LARGE = "FILE_2005"
    
    # API/Inference Errors (3xxx)
    PREDICTION_FAILED = "API_3001"
    SHAP_FAILED = "API_3002"
    MODEL_NOT_READY = "API_3003"
    MODEL_LOAD_ERROR = "API_3004"
    INFERENCE_TIMEOUT = "API_3005"
    
    # Auth Errors (4xxx)
    AUTH_TOKEN_EXPIRED = "AUTH_4001"
    AUTH_TOKEN_INVALID = "AUTH_4002"
    AUTH_CREDENTIALS_INVALID = "AUTH_4003"
    AUTH_USER_NOT_FOUND = "AUTH_4004"
    AUTH_UNAUTHORIZED = "AUTH_4005"
    
    # Database Errors (5xxx)
    DB_CONNECTION_ERROR = "DB_5001"
    DB_QUERY_ERROR = "DB_5002"
    DB_NOT_FOUND = "DB_5003"
    DB_DUPLICATE_ENTRY = "DB_5004"


# User-friendly default messages for each error code
ERROR_MESSAGES = {
    # Validation
    ErrorCode.VALIDATION_MISSING_COLUMNS: "Missing required columns in the uploaded data.",
    ErrorCode.VALIDATION_INVALID_FORMAT: "The file format is invalid.",
    ErrorCode.VALIDATION_EMPTY_VALUES: "Some required fields have missing or empty values.",
    ErrorCode.VALIDATION_INVALID_JSON: "Invalid JSON format.",
    ErrorCode.VALIDATION_INVALID_DATA: "The data contains invalid values.",
    ErrorCode.VALIDATION_MISSING_FIELD: "A required field is missing.",
    
    # File
    ErrorCode.FILE_UNSUPPORTED_TYPE: "This file type is not supported.",
    ErrorCode.FILE_PARSE_ERROR: "Failed to read the file contents.",
    ErrorCode.FILE_UPLOAD_FAILED: "File upload failed.",
    ErrorCode.FILE_NOT_FOUND: "The requested file was not found.",
    ErrorCode.FILE_TOO_LARGE: "The file exceeds the maximum allowed size.",
    
    # API/Inference
    ErrorCode.PREDICTION_FAILED: "Prediction failed. Please try again.",
    ErrorCode.SHAP_FAILED: "Unable to generate SHAP explanations.",
    ErrorCode.MODEL_NOT_READY: "The model is not ready. Please try again later.",
    ErrorCode.MODEL_LOAD_ERROR: "Failed to load the model.",
    ErrorCode.INFERENCE_TIMEOUT: "The inference request timed out.",
    
    # Auth
    ErrorCode.AUTH_TOKEN_EXPIRED: "Your session has expired. Please log in again.",
    ErrorCode.AUTH_TOKEN_INVALID: "Invalid authentication token.",
    ErrorCode.AUTH_CREDENTIALS_INVALID: "Invalid email or password.",
    ErrorCode.AUTH_USER_NOT_FOUND: "User not found.",
    ErrorCode.AUTH_UNAUTHORIZED: "You are not authorized to perform this action.",
    
    # Database
    ErrorCode.DB_CONNECTION_ERROR: "Database connection error.",
    ErrorCode.DB_QUERY_ERROR: "A database error occurred.",
    ErrorCode.DB_NOT_FOUND: "The requested resource was not found.",
    ErrorCode.DB_DUPLICATE_ENTRY: "This entry already exists.",
}


def get_error_message(code: ErrorCode) -> str:
    """Get the default user-friendly message for an error code."""
    return ERROR_MESSAGES.get(code, "An unexpected error occurred.")
