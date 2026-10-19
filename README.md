# ⚙️ TensoScan Backend

API del ecosistema TensoScan: autenticación, asistente y mediciones confirmadas por el usuario (spec B-002). El reconocimiento de los dígitos del tensiómetro **ya no ocurre aquí**: la app lo hace en el
dispositivo y solo envía valores confirmados (ver `specs/`).

## Arquitectura

```
api (routers FastAPI, schemas HTTP)  ─┐
                                      ├─► application (casos de uso) ─► domain (entidades, errores, puertos)
infrastructure (Mongo, JWT, Argon2, LLM) ─┘                                  ▲
                     └──────────────── implementa los puertos ───────────────┘
```

- `app/domain`: reglas y puertos, sin frameworks. `app/application`: casos de uso.
- `app/infrastructure`: MongoDB (PyMongo async), PyJWT, Argon2id, cliente del proveedor LLM e informes PDF (ReportLab).
- `app/api`: routers `v1` y alias `legacy` (cabecera `Deprecation`) para la versión anterior de la app.
- `app/main.py`: `create_app()` (factory); nada pesado se ejecuta al importar.
- Todos los errores usan el sobre `{"error": {"code", "message", "details"}}`.

La especificación, las decisiones (ADR) y los contratos compartidos con la app están en [`specs/`](specs/README.md).

## Puesta en marcha

```bash
cp .env.example .env            # rellena TENSOSCAN_SECRET_KEY (≥ 32 bytes) y las credenciales de Mongo
docker compose up --build       # API en http://127.0.0.1:8000 · documentación en /docs
```

Sin Docker: `uv sync --dev` y `uv run uvicorn app.main:create_app --factory --reload`
(requiere un MongoDB accesible en `TENSOSCAN_MONGODB_URL`).

El servicio **no arranca sin `TENSOSCAN_SECRET_KEY`**. El OCR antiguo del servidor está desactivado por defecto
(`TENSOSCAN_ENABLE_LEGACY_OCR`, necesita `uv sync --extra legacy-ocr` y el modelo en `ia_models/`).

## Desarrollo

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

Los tests de integración usan un MongoDB real si `TENSOSCAN_TEST_MONGODB_URL` está definida (así corren en CI) y un
doble en memoria en caso contrario. El informe PDF se verifica leyendo el PDF generado con `pypdf`.

## Endpoints (v1)

| Método | Ruta | Descripción | Auth |
|--------|------|-------------|------|
| POST | `/v1/auth/register` | Alta de usuario | No |
| POST | `/v1/auth/login` | Access token (30 min) + refresh token (rota en cada uso) | No |
| POST | `/v1/auth/refresh` | Renueva la sesión; reutilizar un refresh ya usado revoca toda la familia | No |
| POST | `/v1/auth/logout` | Cierra la sesión | No |
| GET | `/v1/users/me` | Perfil | Sí |
| DELETE | `/v1/users/me` | Elimina la cuenta y todos sus datos (pide la contraseña) | Sí |
| POST | `/v1/measurements` | Crea una medición confirmada: `201` nueva, `200` reintento idéntico (idempotente por `client_id`), `409` mismo `client_id` con otro contenido, `422` si incumple `reading-rules.md` | Sí |
| GET | `/v1/measurements` | Historial propio, más reciente primero; `from`, `to`, `limit` (1-200, 50 por defecto) y `cursor` | Sí |
| DELETE | `/v1/measurements/{id}` | Borra una medición propia (`404` si no existe o es de otro usuario) | Sí |
| GET | `/v1/measurements/report?format=pdf` | Informe PDF (cabecera, resumen, gráfico, tabla y aviso orientativo); `from`, `to` y `tz` (zona IANA, `Europe/Madrid` por defecto) | Sí |
| POST | `/v1/assistant/messages` | Asistente informativo (no diagnostica) | Sí |
| GET | `/health` | Estado del servicio y de la base de datos | No |

Contrato completo: [`specs/contracts/api-v1.md`](specs/contracts/api-v1.md), con ejemplos reales de petición y
respuesta en [`specs/contracts/examples/`](specs/contracts/examples/) (los regenera
`TENSOSCAN_UPDATE_EXAMPLES=1 uv run pytest tests/contract/test_api_examples.py` y el CI comprueba que no se desvían).

### Mediciones

- Validación: exactamente [`reading-rules.md`](specs/contracts/reading-rules.md), probada con los casos dorados
  (`reading-golden-cases.json`, que se leen desde `specs/contracts/` y no se copian).
- Idempotencia: índice único `(user_id, client_id)` + hash del payload canónico. Dos peticiones simultáneas iguales
  devuelven `201` y `200`.
- Informe: un solo formato, PDF (ReportLab, sin dependencias de sistema). El HTML con JavaScript se eliminó.
- Convivencia legacy: las rutas `/ocr/measurements`, `/ocr/remove/measurements/{id}` y `/ocr/user/measurements/pdf|html`
  (y `POST /ocr/display-recognize` si `TENSOSCAN_ENABLE_LEGACY_OCR=true`) leen y escriben la colección `measurements` con
  el formato antiguo. Cada petición a una ruta legacy deja una línea de log `Legacy route used: <ruta> by user <id>` para
  medir su uso (REQ-B002-13/14).

### Migración de datos antiguos

Copia `recognition_results` → `measurements` sin modificar la colección original. Es idempotente (se puede repetir):

```bash
uv run python -m app.infrastructure.mongo.migrations.m001_measurements --dry-run     # solo informa
uv run python -m app.infrastructure.mongo.migrations.m001_measurements               # migra
uv run python -m app.infrastructure.mongo.migrations.m001_measurements --legacy-tz Europe/Madrid
```

Necesita las mismas variables de entorno que el servicio (`TENSOSCAN_MONGODB_URL`, `TENSOSCAN_SECRET_KEY`). Los
`timestamp` antiguos (sin zona) se interpretan en `--legacy-tz` (por defecto `Europe/Madrid`); la hora repetida del cambio
de otoño usa su primera ocurrencia y la inexistente de primavera se adelanta 1 h. El informe (migradas, ya migradas,
documentos omitidos con su motivo, `ambiguous_local_times` y `nonexistent_local_times`) se imprime, se registra en el log
y, si no es `--dry-run`, se guarda en la colección `migration_reports`. Los valores no numéricos (`"12O"`, `""`) o fuera
de rango (`"300"`) no se migran.

## Licencia

MIT
