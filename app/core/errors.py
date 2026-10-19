"""The only place where domain errors become HTTP responses (error envelope of ``api-v1.md`` §1.1)."""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import current_request_id
from app.domain import errors as domain

logger = logging.getLogger(__name__)

STATUS_BY_CODE: dict[str, int] = {
    domain.NotFound.code: HTTPStatus.NOT_FOUND,
    domain.Forbidden.code: HTTPStatus.FORBIDDEN,
    domain.Unauthorized.code: HTTPStatus.UNAUTHORIZED,
    domain.TokenExpired.code: HTTPStatus.UNAUTHORIZED,
    domain.InvalidCredentials.code: HTTPStatus.UNAUTHORIZED,
    domain.EmailAlreadyRegistered.code: HTTPStatus.CONFLICT,
    domain.Conflict.code: HTTPStatus.CONFLICT,
    domain.ClientIdConflict.code: HTTPStatus.CONFLICT,
    domain.ValidationFailed.code: HTTPStatus.UNPROCESSABLE_ENTITY,
    domain.MeasurementInvalid.code: HTTPStatus.UNPROCESSABLE_ENTITY,
    domain.RateLimited.code: HTTPStatus.TOO_MANY_REQUESTS,
    domain.UpstreamUnavailable.code: HTTPStatus.SERVICE_UNAVAILABLE,
}

_CODE_BY_STATUS: dict[int, str] = {
    HTTPStatus.BAD_REQUEST: "BAD_REQUEST",
    HTTPStatus.UNAUTHORIZED: "UNAUTHORIZED",
    HTTPStatus.FORBIDDEN: "FORBIDDEN",
    HTTPStatus.NOT_FOUND: "NOT_FOUND",
    HTTPStatus.METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    HTTPStatus.CONFLICT: "CONFLICT",
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: "PAYLOAD_TOO_LARGE",
    HTTPStatus.UNPROCESSABLE_ENTITY: "VALIDATION_ERROR",
    HTTPStatus.TOO_MANY_REQUESTS: "RATE_LIMITED",
    HTTPStatus.SERVICE_UNAVAILABLE: "UPSTREAM_UNAVAILABLE",
}


def error_response(
    status: int,
    code: str,
    message: str,
    details: list[dict[str, str]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message, "details": details or []}}
    return JSONResponse(status_code=status, content=body, headers=headers)


async def _domain_error(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, domain.DomainError)  # noqa: S101 - registered for DomainError only
    status = STATUS_BY_CODE.get(error.code, HTTPStatus.INTERNAL_SERVER_ERROR)
    headers: dict[str, str] = {}
    if isinstance(error, domain.RateLimited):
        headers["Retry-After"] = str(error.retry_after)
    if status == HTTPStatus.UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    details = (
        [{"field": issue.field, "code": issue.code} for issue in error.details]
        if isinstance(error, domain.ValidationFailed)
        else None
    )
    return error_response(status, error.code, error.message, details, headers)


async def _request_validation_error(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)  # noqa: S101 - registered for this exception only
    details = [
        {"field": ".".join(str(part) for part in item["loc"][1:]), "code": str(item["type"])} for item in error.errors()
    ]
    return error_response(HTTPStatus.UNPROCESSABLE_ENTITY, "VALIDATION_ERROR", "The request is not valid", details)


async def _http_exception(_: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, StarletteHTTPException)  # noqa: S101 - registered for this exception only
    code = _CODE_BY_STATUS.get(error.status_code, "HTTP_ERROR")
    headers = dict(error.headers) if error.headers else None
    return error_response(error.status_code, code, str(error.detail), headers=headers)


async def _unhandled(_: Request, error: Exception) -> JSONResponse:
    # The cause goes to the log (with the request id); the client only learns that something went wrong.
    logger.error("Unhandled error", exc_info=error, extra={"request_id": current_request_id()})
    return error_response(HTTPStatus.INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", "Internal server error")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(domain.DomainError, _domain_error)
    app.add_exception_handler(RequestValidationError, _request_validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(Exception, _unhandled)
