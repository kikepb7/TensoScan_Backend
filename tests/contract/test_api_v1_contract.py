"""``specs/contracts/api-v1.md`` and the running API must describe the same endpoints (both ways)."""

import re
from pathlib import Path

import pytest
from fastapi import FastAPI

CONTRACT = Path(__file__).resolve().parents[2] / "specs" / "contracts" / "api-v1.md"
IMPLEMENTED = ("/v1/auth", "/v1/users", "/v1/measurements", "/v1/assistant", "/health")
# Documented in the contract but not built yet (spec B-002, T-B002-05).
PENDING = {("GET", "/v1/measurements/report")}
ENDPOINT = re.compile(r"^#{2,4} `(GET|POST|PUT|PATCH|DELETE) (/[^`\s?]*)", re.MULTILINE)

EXPECTED_SUCCESS_STATUS = {
    ("POST", "/v1/auth/register"): "201",
    ("POST", "/v1/auth/login"): "200",
    ("POST", "/v1/auth/refresh"): "200",
    ("POST", "/v1/auth/logout"): "204",
    ("GET", "/v1/users/me"): "200",
    ("DELETE", "/v1/users/me"): "204",
    ("POST", "/v1/assistant/messages"): "200",
    ("POST", "/v1/measurements"): "201",
    ("GET", "/v1/measurements"): "200",
    ("DELETE", "/v1/measurements/{id}"): "204",
}
# Other documented outcomes that must be declared in OpenAPI too.
EXPECTED_OTHER_STATUS = {
    ("POST", "/v1/measurements"): ("200", "409", "422"),
    ("GET", "/v1/measurements"): ("422",),
    ("DELETE", "/v1/measurements/{id}"): ("404",),
}


def _documented() -> set[tuple[str, str]]:
    found = {(method, path) for method, path in ENDPOINT.findall(CONTRACT.read_text())}
    return {(method, path) for method, path in found if path.startswith(IMPLEMENTED)} - PENDING


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


@pytest.mark.parametrize(("operation", "statuses"), sorted(EXPECTED_OTHER_STATUS.items()))
def test_other_documented_status_codes_are_declared(
    app: FastAPI, operation: tuple[str, str], statuses: tuple[str, ...]
) -> None:
    method, path = operation

    declared = app.openapi()["paths"][path][method.lower()]["responses"]

    assert set(statuses) <= declared.keys()


def test_the_measurement_resource_has_exactly_the_documented_fields(app: FastAPI) -> None:
    schemas = app.openapi()["components"]["schemas"]
    documented = {
        "id",
        "client_id",
        "systolic",
        "diastolic",
        "pulse",
        "measured_at",
        "created_at",
        "source",
        "recognition",
    }
    recognition = {"confidence", "model_version", "edited_fields"}

    assert set(schemas["MeasurementResponse"]["properties"]) == documented
    assert set(schemas["RecognitionSchema"]["properties"]) == recognition


def test_the_create_request_is_the_resource_without_id_and_created_at(app: FastAPI) -> None:
    schemas = app.openapi()["components"]["schemas"]
    expected = set(schemas["MeasurementResponse"]["properties"]) - {"id", "created_at"}

    actual = set(schemas["CreateMeasurementRequest"]["properties"])

    assert actual == expected


def test_the_list_response_is_items_and_next_cursor(app: FastAPI) -> None:
    schema = app.openapi()["components"]["schemas"]["MeasurementPageResponse"]

    assert set(schema["properties"]) == {"items", "next_cursor"}


def test_the_listing_accepts_the_documented_query_parameters(app: FastAPI) -> None:
    parameters = app.openapi()["paths"]["/v1/measurements"]["get"]["parameters"]
    by_name = {parameter["name"]: parameter["schema"] for parameter in parameters}

    assert set(by_name) == {"from", "to", "limit", "cursor"}
    assert (by_name["limit"]["minimum"], by_name["limit"]["maximum"], by_name["limit"]["default"]) == (1, 200, 50)
