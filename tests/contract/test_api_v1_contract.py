"""``specs/contracts/api-v1.md`` and the running API must describe the same endpoints (both ways)."""

import re
from pathlib import Path

import pytest
from fastapi import FastAPI

CONTRACT = Path(__file__).resolve().parents[2] / "specs" / "contracts" / "api-v1.md"
# Prefixes implemented so far; measurements and reports join this list with spec B-002.
IMPLEMENTED = ("/v1/auth", "/v1/users", "/v1/assistant", "/health")
ENDPOINT = re.compile(r"^#{2,4} `(GET|POST|PUT|PATCH|DELETE) (/[^`\s?]*)", re.MULTILINE)

EXPECTED_SUCCESS_STATUS = {
    ("POST", "/v1/auth/register"): "201",
    ("POST", "/v1/auth/login"): "200",
    ("POST", "/v1/auth/refresh"): "200",
    ("POST", "/v1/auth/logout"): "204",
    ("GET", "/v1/users/me"): "200",
    ("DELETE", "/v1/users/me"): "204",
    ("POST", "/v1/assistant/messages"): "200",
}


def _documented() -> set[tuple[str, str]]:
    found = {(method, path) for method, path in ENDPOINT.findall(CONTRACT.read_text())}
    return {(method, path) for method, path in found if path.startswith(IMPLEMENTED)}


def _implemented(app: FastAPI) -> set[tuple[str, str]]:
    schema = app.openapi()
    return {
        (method.upper(), path)
        for path, operations in schema["paths"].items()
        if path.startswith(IMPLEMENTED)
        for method in operations
    }


def test_the_contract_lists_the_endpoints_this_spec_implements() -> None:
    assert _documented(), "no endpoints were parsed from api-v1.md: the heading format changed?"


def test_every_documented_endpoint_exists(app: FastAPI) -> None:
    assert _documented() - _implemented(app) == set()


def test_every_implemented_v1_endpoint_is_documented(app: FastAPI) -> None:
    assert _implemented(app) - _documented() == set()


@pytest.mark.parametrize(("operation", "status"), sorted(EXPECTED_SUCCESS_STATUS.items()))
def test_success_status_codes_match_the_contract(app: FastAPI, operation: tuple[str, str], status: str) -> None:
    method, path = operation

    declared = app.openapi()["paths"][path][method.lower()]["responses"]

    assert status in declared


def test_swagger_authenticates_against_the_real_login_route(app: FastAPI) -> None:
    schemes = app.openapi()["components"]["securitySchemes"]

    assert schemes["OAuth2PasswordBearer"]["flows"]["password"]["tokenUrl"] == "/v1/auth/login"
