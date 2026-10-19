from collections.abc import Sequence

import pytest

from app.application.assistant.answer_message import AnswerAssistantMessage
from app.application.assistant.prompts import SYSTEM_PROMPT
from app.domain.assistant import ChatMessage, ChatRole
from app.domain.errors import ValidationFailed


class RecordingLlm:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def complete(self, system_prompt: str, messages: Sequence[ChatMessage]) -> str:
        self.calls.append((system_prompt, list(messages)))
        return "answer"


async def test_the_prompt_is_appended_to_the_history_with_the_safety_system_prompt() -> None:
    llm = RecordingLlm()
    history = [ChatMessage(ChatRole.USER, "hi"), ChatMessage(ChatRole.ASSISTANT, "hello")]

    answer = await AnswerAssistantMessage(llm).execute(prompt="  what is 135/85?  ", history=history)

    assert answer == "answer"
    assert llm.calls == [(SYSTEM_PROMPT, [*history, ChatMessage(ChatRole.USER, "what is 135/85?")])]


async def test_the_safety_prompt_tells_the_assistant_not_to_diagnose_and_to_escalate_emergencies() -> None:
    assert "never diagnose" in SYSTEM_PROMPT
    assert "180/120" in SYSTEM_PROMPT
    assert "urgent medical care" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("prompt", "history_size", "field"),
    [("", 0, "prompt"), ("   ", 0, "prompt"), ("x" * 2001, 0, "prompt"), ("ok", 21, "history")],
)
async def test_invalid_input_is_rejected_before_calling_the_provider(
    prompt: str, history_size: int, field: str
) -> None:
    llm = RecordingLlm()
    history = [ChatMessage(ChatRole.USER, "m")] * history_size

    with pytest.raises(ValidationFailed) as error:
        await AnswerAssistantMessage(llm).execute(prompt=prompt, history=history)

    assert error.value.details[0].field == field
    assert llm.calls == []


async def test_an_overlong_history_message_is_rejected() -> None:
    with pytest.raises(ValidationFailed) as error:
        await AnswerAssistantMessage(RecordingLlm()).execute(
            prompt="ok", history=[ChatMessage(ChatRole.USER, "x" * 2001)]
        )

    assert error.value.details[0].code == "MESSAGE_TOO_LONG"
