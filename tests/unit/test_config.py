import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"mongodb_url": "mongodb://x", "secret_key": "k" * 32}
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_the_service_refuses_to_start_without_a_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENSOSCAN_SECRET_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, mongodb_url="mongodb://x")  # type: ignore[call-arg]


def test_a_short_secret_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        _settings(secret_key="short")


def test_the_defaults_match_the_documented_ones() -> None:
    settings = _settings()

    assert settings.access_token_ttl_minutes == 30
    assert settings.refresh_token_ttl_days == 30
    assert settings.enable_legacy_ocr is False
    assert settings.default_report_tz == "Europe/Madrid"


def test_values_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENSOSCAN_MONGODB_URL", "mongodb://from-env")
    monkeypatch.setenv("TENSOSCAN_SECRET_KEY", "e" * 40)
    monkeypatch.setenv("TENSOSCAN_ACCESS_TOKEN_TTL_MINUTES", "5")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.mongodb_url.get_secret_value() == "mongodb://from-env"
    assert settings.access_token_ttl_minutes == 5


def test_secrets_are_not_printed() -> None:
    assert "kkkk" not in repr(_settings())
