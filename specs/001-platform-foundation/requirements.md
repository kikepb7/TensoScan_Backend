# B-001 — Fundación de plataforma · Requisitos

> Estado: 📝 Pendiente de aprobación · Tipo: Refactor + seguridad
> Resuelve: AUD-B-07 (aislamiento), 09, 10, 12…14, 19, 20, 24…28, 30…45 · Contrato: `api-v1.md` §1, 2, 4 y 5

## 1. Contexto

B-002 (API de mediciones) necesita un servicio que arranque siempre (aunque falte el modelo Keras),
con errores coherentes, sesión segura con refresh, tests contra Mongo y CI. Además hay riesgos de
seguridad críticos (AUD-B-27…29) que no deben esperar a la feature.

## 2. Historias

- **HU-B001-1**: Como operador, quiero que el servicio arranque y responda `/health` aunque falte
  un artefacto opcional, para que una pieza no tumbe todo.
- **HU-B001-2**: Como usuario de la app, quiero que mi sesión no caduque cada 15 minutos sin
  remedio.
- **HU-B001-3**: Como desarrollador, quiero tests, lint, tipos y CI para cambiar el código con
  confianza.
- **HU-B001-4**: Como responsable de seguridad, quiero que no haya secretos por defecto ni en la
  imagen, y que el login resista la fuerza bruta.

## 3. Requisitos (EARS)

### Plataforma
- **REQ-B001-01**: El proyecto DEBERÁ usar Python 3.12 con `pyproject.toml` y `uv.lock`. Las
  dependencias del servicio web DEBERÁN estar separadas de las de `ml/` y de las del OCR legacy
  (`extra` opcional `legacy-ocr`).
- **REQ-B001-02**: La configuración DEBERÁ definirse con `pydantic-settings` (prefijo
  `TENSOSCAN_`, lectura de `.env` solo en desarrollo). El arranque DEBERÁ fallar SI falta
  `SECRET_KEY` o tiene menos de 32 bytes.
- **REQ-B001-03**: La app DEBERÁ construirse con una *factory* `create_app(settings)` y un
  `lifespan` que abra y cierre el cliente Mongo y asegure los índices. Ningún módulo DEBERÁ crear
  clientes ni cargar modelos al importarse.
- **REQ-B001-04**: El OCR legacy DEBERÁ estar desactivado por defecto (`ENABLE_LEGACY_OCR=false`).
  Cuando esté activo, el modelo DEBERÁ cargarse de forma perezosa en la primera petición, y SI no
  puede cargarse, ENTONCES esas rutas DEBERÁN responder `503 UPSTREAM_UNAVAILABLE` sin afectar al
  resto.
- **REQ-B001-05**: El OCR legacy (mientras exista) DEBERÁ trabajar en memoria, sin escribir en
  disco, y exigir autenticación también en `/ocr/recognize`. Las subidas tendrán un límite de 5 MB.
- **REQ-B001-06**: `GET /health` DEBERÁ responder `{status, version}` sin autenticación y
  verificar con un `ping` que Mongo es accesible (`503` si no lo es).
- **REQ-B001-07**: El logging DEBERÁ ser estructurado (JSON en producción) e incluir un
  `request_id` por petición (cabecera `X-Request-ID`, que se propaga o se genera), sin datos
  personales (C-SEC-03).

### Errores
- **REQ-B001-08**: Todas las respuestas de error DEBERÁN seguir el sobre del contrato §1.1.
  Habrá *handlers* para las excepciones de dominio, `RequestValidationError` (→ 422
  `VALIDATION_ERROR` con `details` por campo), `HTTPException` y `Exception` (→ 500 sin detalle).
- **REQ-B001-09**: Ningún router NI caso de uso DEBERÁ capturar `Exception` de forma genérica para
  convertirla en un HTTP. Solo el *handler* global lo hace.

### Arquitectura
- **REQ-B001-10**: El código DEBERÁ organizarse en `domain`, `application`, `infrastructure`,
  `api` y `core` según C-ARCH-01, con puertos (`Protocol`) para usuarios, refresh tokens,
  mediciones (B-002), hashing, tokens y LLM.
- **REQ-B001-11**: Una prueba automática (`import-linter` o un test propio que analice los
  imports) DEBERÁ fallar si se viola la dirección de dependencias.

### Autenticación
- **REQ-B001-12**: Los endpoints de autenticación DEBERÁN implementar `api-v1.md` §2: registro
  (201/409/422), login (OAuth2 *form*, con access y refresh), refresh con **rotación** y detección
  de reutilización (si se reutiliza un refresh revocado, se revoca toda la familia), logout y
  `/v1/users/me`.
- **REQ-B001-13**: Los access tokens DEBERÁN ser JWT HS256 firmados con PyJWT, con `sub`, `iat`,
  `exp` (30 min, configurable), `jti` y `type=access`, y con fechas *aware* en UTC.
- **REQ-B001-14**: Los refresh tokens DEBERÁN ser opacos (`secrets.token_urlsafe(48)`), guardarse
  **hasheados** (SHA-256) con `family_id`, `expires_at` (30 días, configurable) y `revoked_at`, con
  un índice TTL para la limpieza.
- **REQ-B001-15**: CUANDO el token esté caducado, la API DEBERÁ responder `401` con `code =
  TOKEN_EXPIRED`. Si es inválido por cualquier otro motivo (firma, formato o `sub` no válido),
  responderá `401 UNAUTHORIZED`, nunca 500.
- **REQ-B001-16**: Las contraseñas nuevas DEBERÁN hashearse con Argon2id. CUANDO un usuario con
  hash bcrypt legacy inicie sesión correctamente, su hash DEBERÁ re-hashearse con Argon2id.
- **REQ-B001-17**: El email DEBERÁ normalizarse (`strip` + minúsculas) y ser único mediante un
  índice único en Mongo. La carrera de registros simultáneos DEBERÁ resolverse como 409.
- **REQ-B001-18**: DEBERÁ haber rate limit por IP y por email en el login (5 intentos/min), por IP
  en el registro (10/h) y en el refresh (30/min), y por usuario en el asistente (20/min), con
  respuesta `429 RATE_LIMITED` y cabecera `Retry-After`.

### Supresión de cuenta
- **REQ-B001-25**: `DELETE /v1/users/me` DEBERÁ exigir la contraseña y, si es correcta, borrar en
  este orden: mediciones, refresh tokens y usuario. SI el borrado queda a medias por un fallo,
  ENTONCES DEBERÁ poder reintentarse sin dejar datos huérfanos (se borran primero los datos
  dependientes y el usuario al final). Debe registrarse un evento de auditoría sin datos
  personales (`user_deleted`, id con hash).

### Asistente
- **REQ-B001-19**: `POST /v1/assistant/messages` DEBERÁ delegar en un puerto `LlmClient` con el
  modelo configurable (`ASSISTANT_MODEL`), un *system prompt* de seguridad (no diagnosticar,
  derivar a profesionales ante valores de crisis o síntomas) y límites de tamaño (contrato §4). Los
  errores del proveedor DEBERÁN mapearse a `503 UPSTREAM_UNAVAILABLE`.

### Compatibilidad
- **REQ-B001-20**: Las rutas legacy `/register`, `/login`, `/me` y `/ai/chatbot` DEBERÁN seguir
  funcionando (mismos campos de petición y respuesta) delegando en los mismos casos de uso, con la
  cabecera `Deprecation: true` y un `Link` a la ruta v1.

### Operación y calidad
- **REQ-B001-21**: La imagen Docker DEBERÁ ser *multi-stage*, basada en `python:3.12-slim`,
  ejecutarse con un usuario sin privilegios, excluir `.env`, `ml/`, `tests/` y `specs/` y declarar
  `HEALTHCHECK`.
- **REQ-B001-22**: `docker-compose.yaml` (desarrollo) DEBERÁ arrancar Mongo con usuario y
  contraseña desde `.env`, sin publicar el puerto en todas las interfaces (solo `127.0.0.1`) y
  con un volumen con nombre.
- **REQ-B001-23**: CI (GitHub Actions) DEBERÁ ejecutar en cada PR `ruff`, `mypy`, `pytest` con un
  Mongo de servicio, el *gate* de cobertura de C-QA-03 y `docker build`.
- **REQ-B001-24**: DEBERÁ existir `.env.example` con todas las variables y su descripción, sin
  valores secretos.

## 4. Criterios de aceptación

| ID | Escenario | Req. |
|----|-----------|------|
| CA-B001-01 | Sin `ia_models/`, cuando arranco el servicio, entonces `/health` → 200 y `/v1/auth/login` funciona | 03, 04, 06 |
| CA-B001-02 | Sin `SECRET_KEY`, cuando arranco, entonces el proceso termina con un error claro | 02 |
| CA-B001-03 | Dado un borrado de una medición inexistente (legacy), cuando llamo, entonces → 404 con el sobre (no 500) | 08, 09 |
| CA-B001-04 | Dado un JWT con `sub` no válido, cuando llamo a `/v1/users/me`, entonces → 401 `UNAUTHORIZED` | 15 |
| CA-B001-05 | Dado un access token caducado, cuando llamo, entonces → 401 `TOKEN_EXPIRED` | 15 |
| CA-B001-06 | Dado un refresh ya rotado, cuando se reutiliza, entonces → 401 y el refresh vigente de esa familia también queda revocado | 12 |
| CA-B001-07 | Dados dos registros simultáneos con el mismo email, cuando se procesan, entonces uno → 201 y el otro → 409 | 17 |
| CA-B001-08 | Dados 6 logins fallidos en un minuto, cuando hago el sexto, entonces → 429 con `Retry-After` | 18 |
| CA-B001-09 | Dado un usuario con hash bcrypt, cuando inicia sesión, entonces entra y su hash pasa a ser Argon2id | 16 |
| CA-B001-10 | Dada la imagen Docker, cuando la inspecciono, entonces no hay `.env` y el usuario no es root | 21 |
| CA-B001-11 | Dada la app actual (`master`), cuando usa `/login`, `/register`, `/me` y `/ai/chatbot`, entonces funciona igual | 20 |
| CA-B001-13 | Dado un usuario con mediciones y sesiones, cuando elimina su cuenta con la contraseña correcta, entonces → 204 y no queda ningún documento suyo en `users`, `measurements` ni `refresh_tokens`; con contraseña incorrecta → 401 y no se borra nada | 25 |
| CA-B001-12 | Dado un PR, cuando se abre, entonces CI ejecuta lint, tipos, tests con Mongo, cobertura y build | 23 |

## 5. Fuera de alcance
- Recuperación de contraseña y verificación de email.
- Observabilidad avanzada (métricas Prometheus, trazas). Se prepara el `request_id`.
- Despliegue (infraestructura cloud). Q-B001-01.

## 6. Preguntas abiertas

| ID | Pregunta | Propuesta |
|----|----------|-----------|
| Q-B001-01 | ¿Dónde se despliega producción (VPS, Render, Fly, GCP...)? ¿Mongo gestionado (Atlas)? | Afecta solo a la documentación de despliegue y a los secretos; el diseño es agnóstico |
| Q-LEGAL-01 | Valoración regulatoria (RGPD y producto sanitario) y alojamiento en la UE | Imprescindible antes de publicar (ver constitución de la app, C-LEGAL); no bloquea el desarrollo |
| Q-B001-02 | ¿Se mantiene OpenAI como proveedor del asistente? | Sí, detrás del puerto `LlmClient` (intercambiable) |
