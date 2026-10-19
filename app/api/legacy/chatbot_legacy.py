from fastapi import APIRouter, Depends

from app.api.deps import ContainerDep, CurrentUser
from app.api.legacy.headers import deprecated
from app.api.legacy.usage import record_legacy_usage
from app.api.schemas.assistant import AssistantRequest, AssistantResponse

router = APIRouter(tags=["legacy"])


@router.post(
    "/ai/chatbot",
    response_model=AssistantResponse,
    dependencies=[Depends(deprecated("/v1/assistant/messages")), Depends(record_legacy_usage)],
)
async def respond_to_prompt(body: AssistantRequest, user: CurrentUser, container: ContainerDep) -> AssistantResponse:
    container.limiters.assistant.check(user.id)
    answer = await container.answer_assistant.execute(
        prompt=body.prompt,
        history=[message.to_domain() for message in body.history or []],
    )
    return AssistantResponse(response=answer)
