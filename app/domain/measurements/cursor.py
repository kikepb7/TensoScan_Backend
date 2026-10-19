"""Opaque pagination cursor: base64url(json({"m": measured_at, "i": id})) (design §4)."""

import base64
import binascii
import json
import re
from dataclasses import dataclass
from datetime import datetime

from app.domain.errors import FieldIssue, ValidationFailed
from app.domain.measurements.model import as_utc

_OBJECT_ID = re.compile(r"^[0-9a-f]{24}$")


def is_valid_measurement_id(value: str) -> bool:
    return bool(_OBJECT_ID.fullmatch(value))


def _invalid() -> ValidationFailed:
    return ValidationFailed([FieldIssue("cursor", "INVALID")], "The pagination cursor is not valid")


@dataclass(frozen=True)
class PageCursor:
    """Position of the last item of a page: the next page starts strictly after it."""

    measured_at: datetime
    id: str

    def encode(self) -> str:
        raw = json.dumps({"m": as_utc(self.measured_at).isoformat(), "i": self.id}, separators=(",", ":"))
        return base64.urlsafe_b64encode(raw.encode("ascii")).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, token: str) -> "PageCursor":
        """Raises ``ValidationFailed`` (field ``cursor``) for anything that is not a cursor issued by ``encode``."""
        try:
            padded = token + "=" * (-len(token) % 4)
            data = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True).decode("ascii"))
            measured_at = datetime.fromisoformat(data["m"])
            identifier = data["i"]
        except (binascii.Error, ValueError, KeyError, TypeError) as error:
            raise _invalid() from error
        if measured_at.tzinfo is None or not isinstance(identifier, str) or not is_valid_measurement_id(identifier):
            raise _invalid()
        return cls(as_utc(measured_at), identifier)
