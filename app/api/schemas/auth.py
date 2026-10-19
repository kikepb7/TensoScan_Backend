from pydantic import BaseModel, EmailStr, Field

from app.domain.auth import TokenPair
from app.domain.user import User


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    last_name: str

    @classmethod
    def from_domain(cls, user: User) -> "UserResponse":
        return cls(id=user.id, email=user.email, name=user.name, last_name=user.last_name)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

    @classmethod
    def from_domain(cls, pair: TokenPair) -> "TokenResponse":
        return cls(
            access_token=pair.access_token,
            refresh_token=pair.refresh_token,
            token_type=pair.token_type,
            expires_in=pair.expires_in,
        )


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)
