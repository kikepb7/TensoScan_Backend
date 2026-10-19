from pydantic import BaseModel, Field

from app.domain.assistant import ChatMessage, ChatRole


class ChatMessageSchema(BaseModel):
    role: ChatRole
    content: str

    def to_domain(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


class AssistantRequest(BaseModel):
    prompt: str
    history: list[ChatMessageSchema] | None = Field(default=None)


class AssistantResponse(BaseModel):
    response: str
