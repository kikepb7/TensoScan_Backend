from dataclasses import dataclass
from datetime import datetime


def normalize_email(email: str) -> str:
    """Emails are compared case-insensitively and without surrounding whitespace."""
    return email.strip().lower()


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str
    last_name: str
    password_hash: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class NewUser:
    email: str
    name: str
    last_name: str
    password_hash: str
