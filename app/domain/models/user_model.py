from pydantic import BaseModel, EmailStr

class User(BaseModel):
    id: str
    email: EmailStr

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    password: str