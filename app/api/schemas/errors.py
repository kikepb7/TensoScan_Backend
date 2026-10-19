from pydantic import BaseModel


class ErrorDetail(BaseModel):
    field: str
    code: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail]


class ErrorResponse(BaseModel):
    """Error envelope of ``api-v1.md`` §1.1, declared so that OpenAPI documents the 4xx responses."""

    error: ErrorBody
