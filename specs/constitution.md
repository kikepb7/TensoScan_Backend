# Constitución técnica — TensoScan Backend

Principios no negociables. Una spec que necesite romperlos debe modificar antes este documento,
con la justificación correspondiente.

## 1. Arquitectura

**C-ARCH-01. Capas y dirección de dependencias** (puertos y adaptadores):

```
api (routers FastAPI, schemas HTTP, dependencias)  ─┐
                                                     ├─► application (casos de uso) ─► domain (modelos, reglas, puertos)
infrastructure (Mongo, JWT, hashing, PDF, LLM) ─────┘                                   ▲
                       └──────────────── implementa los puertos de domain ──────────────┘
```

| Capa | Puede importar | Prohibido |
|------|----------------|-----------|
| `app/domain` | stdlib, `pydantic` (solo modelos de valor) | FastAPI, Motor/PyMongo, JWT, OpenAI |
| `app/application` | `domain` | FastAPI, drivers |
| `app/infrastructure` | `domain`, `application` (puertos), drivers | `api` |
| `app/api` | `application`, `domain` (tipos), `core` | Acceso directo a colecciones Mongo |
| `app/core` | stdlib, pydantic-settings | — |

**C-ARCH-02.** Los routers **no** contienen lógica de negocio ni acceso a datos: validan la
entrada (schemas), llaman a un caso de uso y serializan la salida.

**C-ARCH-03.** Las dependencias se resuelven con `Depends` desde un *composition root*
(`app/api/deps.py`). No hay estado global creado al importar un módulo, como modelos cargados o
clientes creados en import. Todo se inicializa en el `lifespan`.

**C-ARCH-04.** Los nombres siguen el patrón `<Verbo><Sujeto>` para los casos de uso
(`CreateMeasurement`), `<Entidad>Repository` para los puertos y `Mongo<Entidad>Repository` para
los adaptadores. Se evitan nombres vagos (`utils`, `helpers`, `service` sin calificar).

## 2. Contratos y API

**C-API-01.** Todo endpoint nuevo vive bajo `/v1` y aparece en `specs/contracts/api-v1.md`
**antes** de implementarse.

**C-API-02.** Las respuestas de error siguen el sobre `{error: {code, message, details}}`. Nunca se
devuelven trazas ni `str(exception)` al cliente.

**C-API-03.** Las fechas se guardan y devuelven en UTC con zona (`datetime` *aware*). Queda
prohibido `datetime.now()` sin `tz`.

**C-API-04.** La validación de las mediciones implementa exactamente `reading-rules.md` y se
prueba con `reading-golden-cases.json`.

## 3. Seguridad y privacidad (OWASP API Top 10)

**C-SEC-01.** Los secretos solo se leen del entorno. No existen valores por defecto inseguros: el
arranque falla si falta `SECRET_KEY` o mide menos de 32 bytes.

**C-SEC-02.** Autorización por objeto: toda consulta de datos de usuario filtra por `user_id`
dentro del repositorio. Un recurso ajeno responde 404.

**C-SEC-03.** No se registran contraseñas, tokens, emails completos ni valores de medición. Se
registran ids y códigos.

**C-SEC-04.** Las contraseñas se hashean con Argon2id (o bcrypt como alternativa compatible). Hay
rate limit en login, registro, refresh y asistente.

**C-SEC-05.** El servidor **no** recibe ni guarda imágenes de usuario en v1.

**C-SEC-06.** La imagen Docker no lleva `.env` ni secretos y se ejecuta con un usuario sin
privilegios.

## 4. Calidad

**C-QA-01.** Python 3.12, `pyproject.toml` y lockfile (uv). Formato y lint con `ruff`. `mypy` en
modo estricto para `domain` y `application`.

**C-QA-02.** Tests con `pytest` + `pytest-asyncio` + `httpx.AsyncClient`. Los repositorios Mongo
se prueban contra un Mongo real (contenedor de servicio en CI o `testcontainers` en local). No se
usan *mocks* del driver para la lógica de consultas.

**C-QA-03.** Cobertura mínima de `domain` + `application` ≥ 90 % y global ≥ 80 % (gate en CI).

**C-QA-04.** Tests con nombre `test_<condición>_<resultado>` y estructura *arrange / act /
assert* con las variables `expected` y `actual`.

## 5. ML (`ml/`)

**C-ML-01.** `ml/` es un paquete independiente (`tensoscan_ml`) con sus propias dependencias. El
servicio web **no** importa nada de `ml/`, salvo los fixtures de contrato compartidos.

**C-ML-02.** Reproducibilidad: cada entrenamiento se define con un YAML versionado, una semilla
fija, una versión de dataset con hash y el commit. El artefacto incluye un *model card*.

**C-ML-03.** Los datos reales con personas o dispositivos de usuarios no se versionan en git.
Solo se versiona un manifiesto con hashes. Su uso requiere consentimiento documentado.

**C-ML-04.** Un modelo solo se publica si cumple las métricas de aceptación de B-003 en el set
de test **real**, no solo en el sintético.

## 6. Definition of Done (por tarea)

- [ ] `ruff check`, `ruff format --check`, `mypy` y `pytest` en verde (con la cobertura del
  gate).
- [ ] Los contratos afectados están actualizados (y copiados a la app si cambian).
- [ ] Si hay cambios en la API, la OpenAPI generada (`/openapi.json`) coincide con el contrato
  (test de contrato).
- [ ] La tarea está marcada `[x]` y el commit lleva el formato `T-B00x-yy | ...`.
