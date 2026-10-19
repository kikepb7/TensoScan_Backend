from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import ContainerDep, client_ip
from app.api.schemas.auth import RefreshRequest, RegisterRequest, TokenResponse, UserResponse
from app.domain.user import normalize_email

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def register(body: RegisterRequest, request: Request, container: ContainerDep) -> UserResponse:
    container.limiters.register.check(client_ip(request))
    user = await container.register_user.execute(
        name=body.name, last_name=body.last_name, email=body.email, password=body.password
    )
    return UserResponse.from_domain(user)


@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    container: ContainerDep,
) -> TokenResponse:
    container.limiters.login_by_ip.check(client_ip(request))
    container.limiters.login_by_email.check(normalize_email(form.username))
    pair = await container.authenticate_user.execute(email=form.username, password=form.password)
    return TokenResponse.from_domain(pair)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, request: Request, container: ContainerDep) -> TokenResponse:
    container.limiters.refresh.check(client_ip(request))
    pair = await container.refresh_session.execute(refresh_token=body.refresh_token)
    return TokenResponse.from_domain(pair)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, container: ContainerDep) -> Response:
    await container.logout.execute(refresh_token=body.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
