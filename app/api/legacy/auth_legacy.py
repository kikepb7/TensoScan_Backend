"""Routes of the previous app version, kept (same request and response shapes) until the new app is adopted."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import ContainerDep, CurrentUser, client_ip
from app.api.legacy.headers import deprecated
from app.api.schemas.auth import RegisterRequest, TokenResponse, UserResponse
from app.domain.user import normalize_email

router = APIRouter(tags=["legacy"])


@router.post("/register", dependencies=[Depends(deprecated("/v1/auth/register"))])
async def register(body: RegisterRequest, request: Request, container: ContainerDep) -> dict[str, str]:
    container.limiters.register.check(client_ip(request))
    await container.register_user.execute(
        name=body.name, last_name=body.last_name, email=body.email, password=body.password
    )
    # The key really is "msg:" (with the colon): the previous version returned it that way.
    return {"msg:": "User registered successfully"}


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(deprecated("/v1/auth/login"))])
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    container: ContainerDep,
) -> TokenResponse:
    container.limiters.login_by_ip.check(client_ip(request))
    container.limiters.login_by_email.check(normalize_email(form.username))
    pair = await container.authenticate_user.execute(email=form.username, password=form.password)
    return TokenResponse.from_domain(pair)


@router.get("/me", response_model=UserResponse, dependencies=[Depends(deprecated("/v1/users/me"))])
async def read_me(user: CurrentUser) -> UserResponse:
    return UserResponse.from_domain(user)
