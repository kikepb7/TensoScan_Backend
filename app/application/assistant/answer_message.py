from collections.abc import Sequence

from app.application.assistant.prompts import SYSTEM_PROMPT
from app.domain.assistant import ChatMessage, ChatRole
from app.domain.errors import FieldIssue, ValidationFailed
from app.domain.ports import LlmClient

MAX_HISTORY_MESSAGES = 20
MAX_MESSAGE_LENGTH = 2000


class AnswerAssistantMessage:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    async def execute(self, *, prompt: str, history: Sequence[ChatMessage]) -> str:
        issues = self._validate(prompt, history)
        if issues:
            raise ValidationFailed(issues)
        messages = [*history, ChatMessage(role=ChatRole.USER, content=prompt.strip())]
        return await self._llm.complete(SYSTEM_PROMPT, messages)

    @staticmethod
    def _validate(prompt: str, history: Sequence[ChatMessage]) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        if not prompt.strip():
            issues.append(FieldIssue("prompt", "REQUIRED"))
        elif len(prompt) > MAX_MESSAGE_LENGTH:
            issues.append(FieldIssue("prompt", "TOO_LONG"))
        if len(history) > MAX_HISTORY_MESSAGES:
            issues.append(FieldIssue("history", "TOO_MANY_MESSAGES"))
        if any(len(message.content) > MAX_MESSAGE_LENGTH for message in history):
            issues.append(FieldIssue("history", "MESSAGE_TOO_LONG"))
        return issues
