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

class UserPublic(BaseModel):
    id: str
    email: EmailStr
    name: str
    last_name: str

    @classmethod
    def from_mongo(cls, data: dict):
        return cls(
            id=str(data["_id"]),
            email=data["email"],
            name=data["name"],
            last_name=data["last_name"]
        )