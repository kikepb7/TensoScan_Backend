from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from app.domain.models.user_model import UserCreate, UserPublic
from app.infrastructure.services.auth_service import hash_password, create_access_token, authenticate_user
from app.infrastructure.database.mongo_database import users_collection
from app.infrastructure.services.get_user_service import get_current_user

router = APIRouter()

@router.post("/register")
async def register(user: UserCreate):
    existing = await users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = user.model_dump()
    new_user["hashed_password"] = hash_password(user.password)
    del new_user["password"]
    await users_collection.insert_one(new_user)

    return {"msg:": "User registered successfully"}


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(data={"sub": user.id})

    return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=UserPublic)
async def read_current_user(current_user: dict = Depends(get_current_user)):
    return UserPublic.from_mongo(current_user)