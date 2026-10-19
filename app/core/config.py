"""Application settings, read from the environment (prefix ``TENSOSCAN_``) and an optional ``.env`` file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_SECRET_KEY_BYTES = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TENSOSCAN_", env_file=".env", extra="ignore")

    environment: Literal["dev", "test", "prod"] = "dev"

    mongodb_url: SecretStr
    db_name: str = "tenso_scan_db"

    # JWT signing key. There is deliberately no default: starting without one must fail.
    secret_key: SecretStr
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 30

    assistant_provider_api_key: SecretStr | None = None
    assistant_model: str = "gpt-4o-mini"

    enable_legacy_ocr: bool = False
    legacy_ocr_model_path: Path | None = None

    log_json: bool = True
    default_report_tz: str = "Europe/Madrid"

    # Rate limits (requests per window). Tests tighten or relax them.
    login_attempts_per_minute: int = 5
    register_per_hour: int = 10
    refresh_per_minute: int = 30
    assistant_per_minute: int = 20

    @field_validator("secret_key")
    @classmethod
    def _secret_key_is_strong_enough(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode()) < MIN_SECRET_KEY_BYTES:
            raise ValueError(f"must be at least {MIN_SECRET_KEY_BYTES} bytes long")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()  # required values come from the environment
