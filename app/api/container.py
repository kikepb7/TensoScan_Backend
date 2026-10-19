"""Composition root: builds every collaborator once at start-up and hands them to the routers via ``app.state``."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx

from app.application.assistant.answer_message import AnswerAssistantMessage
from app.application.auth.authenticate_user import AuthenticateUser
from app.application.auth.delete_account import DeleteAccount
from app.application.auth.get_current_user import GetCurrentUser
from app.application.auth.logout import Logout
from app.application.auth.refresh_session import RefreshSession
from app.application.auth.register_user import RegisterUser
from app.application.auth.tokens import SessionIssuer
from app.application.measurements.erase_measurements import EraseAllMeasurements
from app.core.config import Settings
from app.core.rate_limit import RateLimiter
from app.domain.clock import SystemClock
from app.domain.ports import Clock, MeasurementRepository
from app.infrastructure.llm.openai_client import OpenAiLlmClient
from app.infrastructure.mongo.client import Database
from app.infrastructure.mongo.legacy_results_eraser import LegacyResultsEraser
from app.infrastructure.mongo.measurement_repository import MongoMeasurementRepository
from app.infrastructure.mongo.refresh_token_repository import MongoRefreshTokenRepository
from app.infrastructure.mongo.user_repository import MongoUserRepository
from app.infrastructure.security.jwt_service import JwtAccessTokenService
from app.infrastructure.security.password_hasher import PwdlibPasswordHasher

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class RateLimiters:
    login_by_ip: RateLimiter
    login_by_email: RateLimiter
    register: RateLimiter
    refresh: RateLimiter
    assistant: RateLimiter


@dataclass(frozen=True)
class Container:
    settings: Settings
    db: Database
    clock: Clock
    limiters: RateLimiters
    register_user: RegisterUser
    authenticate_user: AuthenticateUser
    refresh_session: RefreshSession
    logout: Logout
    get_current_user: GetCurrentUser
    delete_account: DeleteAccount
    answer_assistant: AnswerAssistantMessage
    measurements: MeasurementRepository
    legacy_ocr: Any = None


def build_container(
    settings: Settings,
    db: Database,
    http: httpx.AsyncClient,
    clock: Clock | None = None,
) -> Container:
    clock = clock or SystemClock()
    users = MongoUserRepository(db)
    refresh_tokens = MongoRefreshTokenRepository(db)
    hasher = PwdlibPasswordHasher()
    access_tokens = JwtAccessTokenService(
        secret_key=settings.secret_key.get_secret_value(),
        ttl=timedelta(minutes=settings.access_token_ttl_minutes),
        clock=clock,
    )
    issuer = SessionIssuer(access_tokens, refresh_tokens, clock, timedelta(days=settings.refresh_token_ttl_days))
    measurements = MongoMeasurementRepository(db)
    # Account deletion erases the new collection and, while it still exists, the legacy one.
    eraser = EraseAllMeasurements([measurements, LegacyResultsEraser(db)])
    return Container(
        settings=settings,
        db=db,
        clock=clock,
        limiters=RateLimiters(
            login_by_ip=RateLimiter(settings.login_attempts_per_minute, SECONDS_PER_MINUTE),
            login_by_email=RateLimiter(settings.login_attempts_per_minute, SECONDS_PER_MINUTE),
            register=RateLimiter(settings.register_per_hour, SECONDS_PER_HOUR),
            refresh=RateLimiter(settings.refresh_per_minute, SECONDS_PER_MINUTE),
            assistant=RateLimiter(settings.assistant_per_minute, SECONDS_PER_MINUTE),
        ),
        register_user=RegisterUser(users, hasher),
        authenticate_user=AuthenticateUser(users, hasher, issuer),
        refresh_session=RefreshSession(users, refresh_tokens, issuer, clock),
        logout=Logout(refresh_tokens, clock),
        get_current_user=GetCurrentUser(users, access_tokens),
        delete_account=DeleteAccount(users, hasher, refresh_tokens, eraser),
        answer_assistant=AnswerAssistantMessage(
            OpenAiLlmClient(http, settings.assistant_provider_api_key, settings.assistant_model)
        ),
        measurements=measurements,
        legacy_ocr=_legacy_ocr(settings),
    )


def _legacy_ocr(settings: Settings) -> Any:
    if not settings.enable_legacy_ocr:
        return None
    from app.api.legacy.ocr_legacy import LegacyOcr

    return LegacyOcr(settings.legacy_ocr_model_path)
