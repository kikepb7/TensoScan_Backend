# ⚙️ TensoScan Backend

API del ecosistema TensoScan: autenticación, asistente y (desde la spec B-002) mediciones confirmadas por el
usuario. El reconocimiento de los dígitos del tensiómetro **ya no ocurre aquí**: la app lo hace en el
dispositivo y solo envía valores confirmados (ver `specs/`).

## Arquitectura

```
api (routers FastAPI, schemas HTTP)  ─┐
                                      ├─► application (casos de uso) ─► domain (entidades, errores, puertos)
infrastructure (Mongo, JWT, Argon2, LLM) ─┘                                  ▲
                     └──────────────── implementa los puertos ───────────────┘
```

- `app/domain`: reglas y puertos, sin frameworks. `app/application`: casos de uso.
- `app/infrastructure`: MongoDB (PyMongo async), PyJWT, Argon2id, cliente del proveedor LLM.
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
doble en memoria en caso contrario.

## Endpoints (v1)

| Método | Ruta | Descripción | Auth |
|--------|------|-------------|------|
| POST | `/v1/auth/register` | Alta de usuario | No |
| POST | `/v1/auth/login` | Access token (30 min) + refresh token (rota en cada uso) | No |
| POST | `/v1/auth/refresh` | Renueva la sesión; reutilizar un refresh ya usado revoca toda la familia | No |
| POST | `/v1/auth/logout` | Cierra la sesión | No |
| GET | `/v1/users/me` | Perfil | Sí |
| DELETE | `/v1/users/me` | Elimina la cuenta y todos sus datos (pide la contraseña) | Sí |
| POST | `/v1/assistant/messages` | Asistente informativo (no diagnostica) | Sí |
| GET | `/health` | Estado del servicio y de la base de datos | No |

Contrato completo: [`specs/contracts/api-v1.md`](specs/contracts/api-v1.md).

## Licencia

MIT
