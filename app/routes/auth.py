from fastapi import APIRouter, HTTPException, status, Depends, Response
from app.schemas.auth import UserCreate, UserResponse, Token, UserLogin
from app.routes.deps import get_database, get_current_user
from app.services.auth_service import AuthService

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register(user_in: UserCreate, db=Depends(get_database)):
    auth_service = AuthService(db)
    return await auth_service.register_user(user_in)

@router.post("/login")
async def login(response: Response, user_in: UserLogin, db=Depends(get_database)):
    auth_service = AuthService(db)
    user = await auth_service.authenticate_user(user_in)
    token_data = auth_service.create_token_for_user(user)
    
    # Set HttpOnly Cookie (Session Cookie - expires on browser close)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {token_data['access_token']}",
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production (HTTPS)
        path="/",  # Important: makes cookie available for all paths
    )
    
    return {"success": True, "user": {"username": user["username"]}}

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"success": True, "message": "Logged out successfully"}

@router.get("/profile")
async def verify_session(current_user: str = Depends(get_current_user)):
    return {"authenticated": True, "username": current_user}

