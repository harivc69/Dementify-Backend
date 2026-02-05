from fastapi import APIRouter, HTTPException, status, Depends
from app.schemas.auth import UserCreate, UserResponse, Token, UserLogin
from app.routes.deps import get_database
from app.services.auth_service import AuthService

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register(user_in: UserCreate, db=Depends(get_database)):
    auth_service = AuthService(db)
    return await auth_service.register_user(user_in)

@router.post("/login", response_model=Token)
async def login(user_in: UserLogin, db=Depends(get_database)):
    auth_service = AuthService(db)
    user = await auth_service.authenticate_user(user_in)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_service.create_token_for_user(user)
