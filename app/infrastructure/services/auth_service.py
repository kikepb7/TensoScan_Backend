from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from app.core.config import settings
from app.infrastructure.database.mongo_database import users_collection
from app.domain.models.user_model import User, UserInDB

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def hash_password(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def authenticate_user(email: str, password: str):
    user = await users_collection.find_one({"email": email})

    if not user or not verify_password(password, user["hashed_password"]):
        return None

    user["id"] = str(user["_id"])
    del user["_id"]
    return UserInDB(**user)