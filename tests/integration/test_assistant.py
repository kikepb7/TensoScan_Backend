import json

import httpx

from tests.conftest import LlmHandler, bearer, register_and_login


async def test_the_assistant_answers_with_the_provider_reply(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)

    response = await client.post("/v1/assistant/messages", headers=bearer(tokens), json={"prompt": "What is 135/85?"})

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the model"}


async def test_the_provider_receives_the_safety_prompt_and_the_conversation(
    client: httpx.AsyncClient, llm_handler: list[LlmHandler]
) -> None:
    sent: list[dict[str, object]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm_handler[0] = capture
    tokens = await register_and_login(client)

    await client.post(
        "/v1/assistant/messages",
        headers=bearer(tokens),
        json={"prompt": "and now?", "history": [{"role": "user", "content": "hi"}]},
    )

    messages = sent[0]["messages"]
    assert messages[0]["role"] == "system" and "never diagnose" in messages[0]["content"]  # type: ignore[index]
    assert [m["role"] for m in messages[1:]] == ["user", "user"]  # type: ignore[index, union-attr]


async def test_a_provider_failure_is_reported_as_service_unavailable_without_leaking_its_body(
    client: httpx.AsyncClient, llm_handler: list[LlmHandler]
) -> None:
    llm_handler[0] = lambda _: httpx.Response(500, text="upstream exploded: key=sk-secret")
    tokens = await register_and_login(client)

    response = await client.post("/v1/assistant/messages", headers=bearer(tokens), json={"prompt": "hi"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    assert "sk-secret" not in response.text


async def test_a_network_failure_towards_the_provider_is_service_unavailable(
    client: httpx.AsyncClient, llm_handler: list[LlmHandler]
) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm_handler[0] = refuse
    tokens = await register_and_login(client)

    response = await client.post("/v1/assistant/messages", headers=bearer(tokens), json={"prompt": "hi"})

    assert response.status_code == 503


async def test_too_long_a_conversation_is_rejected(client: httpx.AsyncClient) -> None:
    tokens = await register_and_login(client)
    history = [{"role": "user", "content": "m"}] * 21

    response = await client.post(
        "/v1/assistant/messages", headers=bearer(tokens), json={"prompt": "hi", "history": history}
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"] == [{"field": "history", "code": "TOO_MANY_MESSAGES"}]


async def test_the_assistant_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/assistant/messages", json={"prompt": "hi"})

    assert response.status_code == 401
