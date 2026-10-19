import logging
from collections.abc import Sequence

import httpx
from pydantic import SecretStr

from app.domain.assistant import ChatMessage
from app.domain.errors import UpstreamUnavailable

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 30.0
HTTP_ERROR_THRESHOLD = 400


class OpenAiLlmClient:
    def __init__(self, http: httpx.AsyncClient, api_key: SecretStr | None, model: str) -> None:
        self._http = http
        self._api_key = api_key
        self._model = model

    async def complete(self, system_prompt: str, messages: Sequence[ChatMessage]) -> str:
        if self._api_key is None:
            logger.error("The assistant provider API key is not configured")
            raise UpstreamUnavailable

        payload = {
            "model": self._model,
            "temperature": 0.4,
            "messages": [
                {"role": "system", "content": system_prompt},
                *({"role": message.role.value, "content": message.content} for message in messages),
            ],
        }
        try:
            response = await self._http.post(
                CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as error:
            logger.warning("Assistant provider unreachable: %s", type(error).__name__)
            raise UpstreamUnavailable from error

        if response.status_code >= HTTP_ERROR_THRESHOLD:
            # Only the status is logged: the body may echo the prompt, which can contain health data.
            logger.warning("Assistant provider answered HTTP %s", response.status_code)
            raise UpstreamUnavailable

        try:
            return str(response.json()["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            logger.warning("Assistant provider returned an unexpected payload")
            raise UpstreamUnavailable from error
