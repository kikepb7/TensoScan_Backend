# B-001 — Fundación de plataforma · Diseño

> Estado: 📝 Pendiente de aprobación · Requisitos: [requirements.md](requirements.md)

## 1. Estructura objetivo

```
TensoScan_Backend/
├── pyproject.toml            # proyecto "tensoscan-api"; extras: dev, legacy-ocr
├── uv.lock
├── .env.example
├── Dockerfile                # multi-stage, no root
├── docker-compose.yaml       # dev: api + mongo (con auth)
├── app/
│   ├── main.py               # create_app() + lifespan
│   ├── core/
│   │   ├── config.py         # Settings (pydantic-settings)
│   │   ├── logging.py        # JSON formatter + middleware de request_id
│   │   ├── errors.py         # handlers → sobre de error
│   │   └── rate_limit.py
│   ├── domain/
│   │   ├── errors.py         # DomainError y subclases (con code y http_status lógico)
│   │   ├── user.py           # User, Email (value object)
│   │   ├── auth.py           # TokenPair, RefreshTokenRecord
│   │   ├── assistant.py      # ChatMessage, ChatRole
│   │   └── ports.py          # UserRepository, RefreshTokenRepository, PasswordHasher, AccessTokenIssuer, LlmClient, Clock
│   ├── application/
│   │   ├── auth/             # RegisterUser, AuthenticateUser, RefreshSession, Logout, GetCurrentUser
│   │   └── assistant/        # AnswerAssistantMessage
│   ├── infrastructure/
│   │   ├── mongo/            # client.py, indexes.py, user_repository.py, refresh_token_repository.py
│   │   ├── security/         # argon2_hasher.py, jwt_issuer.py
│   │   ├── llm/              # openai_client.py (httpx)
│   │   └── legacy_ocr/       # (extra legacy-ocr) código OCR actual aislado, perezoso y en memoria
│   └── api/
│       ├── deps.py           # composition root: get_settings, get_db, repos, casos de uso, current_user
│       ├── v1/               # auth.py, users.py, assistant.py, health.py (+ measurements.py en B-002)
│       ├── legacy/           # auth_legacy.py, chatbot_legacy.py, ocr_legacy.py (+ measurements_legacy.py en B-002)
│       └── schemas/          # modelos Pydantic de petición y respuesta (snake_case)
├── tests/
│   ├── unit/                 # domain + application con dobles en memoria
│   ├── integration/          # api + mongo real
│   ├── contract/             # golden cases + OpenAPI frente a api-v1.md
│   └── conftest.py
├── ml/                       # B-003 (paquete independiente)
└── specs/
```

## 2. Decisiones (ADR)

### ADR-B001-01 — `uv` + `pyproject.toml`
- **Motivo**: lockfile reproducible, instalación rápida en CI y en Docker, y *extras* para
  separar el OCR legacy.
- **Descartado**: `requirements.txt` fijado a mano (estado actual y causa de AUD-B-38) y Poetry
  (más lento, sin ventaja aquí).

### ADR-B001-02 — PyJWT en lugar de python-jose
- **Motivo**: python-jose tiene CVEs publicadas (confusión de algoritmos o DoS con JWE) y poco
  mantenimiento. PyJWT está mantenido, la API es casi idéntica y solo se necesita HS256.

### ADR-B001-03 — Argon2id (`pwdlib`) con verificación bcrypt de compatibilidad
- **Motivo**: passlib no tiene mantenimiento y choca con bcrypt ≥ 4.1. Argon2id es la
  recomendación actual de OWASP.
- **Migración transparente**: `PasswordHasher.verify()` reconoce el prefijo `$2b$` y verifica con
  bcrypt; si es correcto, `needs_rehash=True` → `AuthenticateUser` re-hashea (REQ-B001-16).

### ADR-B001-04 — Refresh tokens opacos con rotación por familia
- **Alternativas**:
  - refresh JWT sin estado: no se puede revocar ni detectar su reutilización;
  - access tokens largos (días): un token robado sirve demasiado tiempo;
  - **opaco + rotación + familia** ✅: revocable, y la reutilización de un token robado invalida
    toda la familia.
- **Modelo** (`refresh_tokens`): `{_id, token_hash (único), user_id, family_id, issued_at,
  expires_at (TTL), revoked_at?, replaced_by?}`.
- **Refresh**: buscar por hash →
  - no existe o está caducado → 401;
  - `revoked_at != null` → **reutilización**: revocar todos los de `family_id` y responder 401;
  - válido → marcar `revoked_at` y `replaced_by` y emitir un par nuevo con el mismo `family_id`.

  La operación es atómica con `find_one_and_update` condicionado a `revoked_at == null`.

### ADR-B001-05 — Driver: PyMongo Async API (y no Motor)
- **Motivo**: MongoDB anunció la deprecación de Motor en favor de la API asíncrona nativa de
  PyMongo (`AsyncMongoClient`). Como hay que actualizar de todos modos (Motor 2.5 y PyMongo 3.12),
  se salta directamente a la API soportada.
- **Riesgo**: cambios menores de API. Los repositorios aíslan el driver (C-ARCH-01).

### ADR-B001-06 — El OCR legacy se aísla en un *extra* y se desactiva por defecto
- El servicio deja de depender de TensorFlow para arrancar (AUD-B-09). El código actual se mueve
  a `infrastructure/legacy_ocr/` sin mejorarlo (se retira en B-002), salvo para:
  - cargar el modelo de forma perezosa;
  - trabajar en memoria (sin `imwrite`, `imshow` ni ficheros temporales);
  - exigir autenticación y limitar el tamaño.
- La imagen de producción se construye **sin** el extra, salvo que se pase
  `--build-arg WITH_LEGACY_OCR=1` durante el periodo de convivencia.

### ADR-B001-07 — Rate limiting en memoria por proceso (v1)
- `limits` con almacenamiento en memoria (o `slowapi`). Con un único proceso y réplica es
  suficiente. Si se escala horizontalmente, se cambia a almacenamiento en Mongo o Redis
  (documentado como deuda).

## 3. Configuración

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TENSOSCAN_", env_file=".env", extra="ignore")
    environment: Literal["dev", "test", "prod"] = "dev"
    mongodb_url: SecretStr
    db_name: str = "tenso_scan_db"
    secret_key: SecretStr                       # obligatorio, len >= 32 (validator)
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 30
    enable_legacy_ocr: bool = False
    legacy_ocr_model_path: Path | None = None
    assistant_provider_api_key: SecretStr | None = None
    assistant_model: str = "gpt-4o-mini"        # configurable; validar disponibilidad al desplegar
    log_json: bool = True
```

## 4. Errores

```python
class DomainError(Exception):
    code: ClassVar[str]
class NotFound(DomainError): code = "NOT_FOUND"
class EmailAlreadyRegistered(DomainError): code = "EMAIL_ALREADY_REGISTERED"
class InvalidCredentials(DomainError): code = "INVALID_CREDENTIALS"
class TokenExpired(DomainError): code = "TOKEN_EXPIRED"
class Unauthorized(DomainError): code = "UNAUTHORIZED"
class ValidationFailed(DomainError): code = "VALIDATION_ERROR"; details: list[FieldIssue]
class Conflict(DomainError): ...
class UpstreamUnavailable(DomainError): code = "UPSTREAM_UNAVAILABLE"
class RateLimited(DomainError): code = "RATE_LIMITED"; retry_after: int
```
`core/errors.py` define la tabla `code → status HTTP` (la del contrato §1.1). Es el **único**
punto donde el dominio se traduce a HTTP.

## 5. Flujo de petición

```
Request → RequestIdMiddleware → RateLimit (dependencia por ruta) → Router (schema) →
  Depends(current_user) → caso de uso (application) → puertos (infrastructure) → Mongo
  ← resultado de dominio ← Router serializa el schema de respuesta
Errores: DomainError → handler → sobre ; Exception → handler → 500 INTERNAL_ERROR (log con request_id)
```

## 6. Asistente

- *System prompt* (versionado en `app/application/assistant/prompts.py`):
  - eres un asistente informativo sobre la tensión arterial;
  - no diagnosticas ni prescribes;
  - ante valores ≥ 180/120 o síntomas (dolor torácico, disnea, déficit neurológico) recomiendas
    atención urgente;
  - respondes en el idioma del usuario.
- El historial se trunca a 20 mensajes y cada mensaje a 2.000 caracteres (si se supera, 422).
- `OpenAiLlmClient` usa httpx con timeout de 30 s. Un 429 o 5xx del proveedor →
  `UpstreamUnavailable`.

## 7. Docker

```dockerfile
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:<versión fijada> /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
ARG WITH_LEGACY_OCR=0
RUN uv sync --frozen --no-dev $( [ "$WITH_LEGACY_OCR" = "1" ] && echo "--extra legacy-ocr" )
COPY app ./app

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 api
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER api
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
```
`.dockerignore`: `.env*` (salvo `.env.example`), `.git`, `ml/`, `tests/`, `specs/`, `data/`,
`ia_models/` y `__pycache__`.

## 8. Testing

- **Unit**: casos de uso con *fakes* en memoria de los puertos (`InMemoryUserRepository`...) y un
  `FakeClock`.
- **Integración**: `httpx.AsyncClient(transport=ASGITransport(app))` contra un Mongo real (servicio
  `mongo:7` en CI y `testcontainers` en local). Cada test usa una base de datos con un nombre
  aleatorio.
- **Contrato**: un test compara `app.openapi()` con los paths, métodos y códigos de
  `api-v1.md` (lista mantenida en `tests/contract/api_v1_expectations.py`).
- **Arquitectura**: test de imports (o `import-linter`) para C-ARCH-01.

## 9. Riesgos

| Riesgo | Mitigación |
|--------|------------|
| Romper la app actual al mover rutas | Alias legacy (REQ-B001-20) + test de compatibilidad con las peticiones exactas que hace la app en `master` |
| Hashes bcrypt legacy incompatibles | Verificación dual + re-hash (ADR-B001-03) |
| Cambio de driver (Motor → PyMongo Async) | Repositorios aislados y tests de integración |
