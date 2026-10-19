"""Domain errors. Only ``app.core.errors`` translates them to HTTP."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class FieldIssue:
    field: str
    code: str


class DomainError(Exception):
    code: ClassVar[str] = "INTERNAL_ERROR"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class NotFound(DomainError):
    code = "NOT_FOUND"


class Forbidden(DomainError):
    code = "FORBIDDEN"


class Unauthorized(DomainError):
    code = "UNAUTHORIZED"


class TokenExpired(DomainError):
    code = "TOKEN_EXPIRED"


class InvalidCredentials(DomainError):
    code = "INVALID_CREDENTIALS"


class EmailAlreadyRegistered(DomainError):
    code = "EMAIL_ALREADY_REGISTERED"


class Conflict(DomainError):
    code = "CONFLICT"


class ClientIdConflict(Conflict):
    code = "CLIENT_ID_CONFLICT"


class UpstreamUnavailable(DomainError):
    code = "UPSTREAM_UNAVAILABLE"


class RateLimited(DomainError):
    code = "RATE_LIMITED"

    def __init__(self, retry_after: int, message: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ValidationFailed(DomainError):
    code = "VALIDATION_ERROR"

    def __init__(self, details: list[FieldIssue] | None = None, message: str | None = None) -> None:
        super().__init__(message)
        self.details: list[FieldIssue] = details or []


class MeasurementInvalid(ValidationFailed):
    code = "MEASUREMENT_INVALID"


__all__ = [
    "ClientIdConflict",
    "Conflict",
    "DomainError",
    "EmailAlreadyRegistered",
    "FieldIssue",
    "Forbidden",
    "InvalidCredentials",
    "MeasurementInvalid",
    "NotFound",
    "RateLimited",
    "TokenExpired",
    "Unauthorized",
    "UpstreamUnavailable",
    "ValidationFailed",
]
