from fastapi import APIRouter, Response, status

from app.api.deps import ContainerDep, CurrentUser
from app.api.schemas.auth import DeleteAccountRequest, UserResponse

router = APIRouter(prefix="/v1/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
async def read_me(user: CurrentUser) -> UserResponse:
    return UserResponse.from_domain(user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_me(body: DeleteAccountRequest, user: CurrentUser, container: ContainerDep) -> Response:
    await container.delete_account.execute(user_id=user.id, password=body.password)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
