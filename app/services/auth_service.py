from datetime import timedelta
from fastapi import status
from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.config import settings
from app.schemas.auth import UserCreate, UserLogin, Token, UserResponse
from app.core.exceptions import ValidationError, AuthError
from app.core.error_codes import ErrorCode


class AuthService:
    def __init__(self, db):
        self.db = db
        self.collection = db.users

    async def authenticate_user(self, user_in: UserLogin) -> dict:
        """Authenticate user and return user doc if successful."""
        user = await self.collection.find_one({"username": user_in.username})
        if not user or not verify_password(user_in.password, user["hashed_password"]):
            raise AuthError(
                message="Incorrect username or password",
                code=ErrorCode.AUTH_CREDENTIALS_INVALID
            )
        return user

    async def register_user(self, user_in: UserCreate) -> dict:
        """Register a new user."""
        # Check existing
        if await self.collection.find_one({"username": user_in.username}):
            raise ValidationError(
                message="Username already registered",
                code=ErrorCode.DB_DUPLICATE_ENTRY
            )
        if await self.collection.find_one({"email": user_in.email}):
            raise ValidationError(
                message="Email already registered",
                code=ErrorCode.DB_DUPLICATE_ENTRY
            )

        
        user_dict = user_in.dict()
        user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))
        
        result = await self.collection.insert_one(user_dict)
        user_dict["_id"] = str(result.inserted_id)
        return user_dict

    def create_token_for_user(self, user: dict) -> Token:
        """Create access token for user."""
        access_token = create_access_token(
            data={"sub": user["username"]}
        )
        return {
            "access_token": access_token, 
            "token_type": "bearer"
        }
