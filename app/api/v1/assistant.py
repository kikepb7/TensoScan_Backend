from fastapi import APIRouter

from app.api.deps import ContainerDep, CurrentUser
from app.api.schemas.assistant import AssistantRequest, AssistantResponse

router = APIRouter(prefix="/v1/assistant", tags=["assistant"])


@router.post("/messages", response_model=AssistantResponse)
async def send_message(body: AssistantRequest, user: CurrentUser, container: ContainerDep) -> AssistantResponse:
    container.limiters.assistant.check(user.id)
    answer = await container.answer_assistant.execute(
        prompt=body.prompt,
        history=[message.to_domain() for message in body.history or []],
    )
    return AssistantResponse(response=answer)
