# B-001 — Fundación de plataforma · Tareas

> Commit: `T-B001-xx | <descripción en inglés>`. Cada tarea cumple la DoD de la constitución.

## Fase 1 — Herramientas y arranque seguro
- [ ] **T-B001-01 · `pyproject.toml` + uv + ruff + mypy + pytest + pre-commit**
  - Python 3.12. Depurar dependencias (se eliminan Plotly, Kaleido, Matplotlib, librerías de PDF
    duplicadas, `pip` y `setuptools` fijados, y TensorFlow/OpenCV/scikit-image, que pasan al
    *extra* `legacy-ocr`).
  - Borrar `requirements.txt` cuando el lock sea equivalente.
  - Hallazgos: AUD-B-38, 39 · Req.: REQ-B001-01.
- [ ] **T-B001-02 · Settings**
  - `pydantic-settings`, `SECRET_KEY` obligatoria con validación y `.env.example`.
  - Tests: falta el secreto, el secreto es corto y la configuración es válida.
  - Hallazgos: AUD-B-27, 43 · Req.: REQ-B001-02, 24.
- [ ] **T-B001-03 · `create_app` + `lifespan` + `/health` + logging**
  - Cliente Mongo en el `lifespan` y `request_id`. OCR legacy perezoso, en memoria, autenticado,
    con límite de 5 MB y tras flag. Se eliminan los `print`, `imshow`, `imwrite` y la escritura de
    `display.jpg`. Se arregla `main.py` (título y raíz).
  - Hallazgos: AUD-B-07, 08 (aislados), 09, 10, 19, 41, 42 · Req.: REQ-B001-03…07.
  - DoD: CA-B001-01.
- [ ] **T-B001-04 · Errores**
  - `DomainError`, *handlers* y sobre de error. Eliminar los `except Exception` de los routers.
  - Hallazgos: AUD-B-12, 13 · Req.: REQ-B001-08, 09. DoD: CA-B001-03.

## Fase 2 — Arquitectura y autenticación
- [ ] **T-B001-05 · Capas + puertos + composition root + test de arquitectura**
  - Mover usuarios a `UserRepository` (Mongo). Sustituir `mongo_database.py` por
    `infrastructure/mongo/client.py`. Índices: `users.email` único.
  - Hallazgos: AUD-B-14, 42 · Req.: REQ-B001-10, 11.
- [ ] **T-B001-06 · Seguridad de tokens y contraseñas**
  - PyJWT, fechas *aware*, `jti` y `type`, 401 `TOKEN_EXPIRED` frente a `UNAUTHORIZED` y
    validación del `sub`. `pwdlib` Argon2id + verificación bcrypt + re-hash.
  - Hallazgos: AUD-B-30, 31, 32, 35 · Req.: REQ-B001-13, 15, 16. DoD: CA-B001-04, 05 y 09.
- [ ] **T-B001-07 · Auth v1 + refresh tokens + `/v1/users/me`**
  - Registro 201/409 (normalización + índice único), login, refresh con rotación y familia,
    logout. Corregir `tokenUrl`. Índice TTL de `refresh_tokens`.
  - Hallazgos: AUD-B-20, 26, 34, 36 · Req.: REQ-B001-12, 14, 17. DoD: CA-B001-06, 07.
- [ ] **T-B001-08 · Rate limiting**
  - Req.: REQ-B001-18 · Hallazgos: AUD-B-33. DoD: CA-B001-08.
- [ ] **T-B001-09 · Asistente v1**
  - Puerto `LlmClient`, *system prompt*, límites y mapeo 503. Eliminar `from openai import BaseModel`.
  - Hallazgos: AUD-B-24, 25 · Req.: REQ-B001-19.
- [ ] **T-B001-12 · Alias legacy de auth, perfil y chatbot con cabecera `Deprecation`**
  - Test de compatibilidad con las peticiones de la app actual.
  - Req.: REQ-B001-20. DoD: CA-B001-11.

## Fase 3 — Operación
- [ ] **T-B001-10 · Docker y compose seguros**
  - Hallazgos: AUD-B-28, 29, 37, 45 · Req.: REQ-B001-21, 22. DoD: CA-B001-10.
- [ ] **T-B001-11 · CI**
  - GitHub Actions con lint, tipos, tests con el servicio Mongo, cobertura y `docker build`.
    Mover la plantilla de PR a `.github/pull_request_template.md` (secciones: Spec/Tareas,
    Cambios, Tests, Checklist DoD).
  - Hallazgos: AUD-B-39, 40 · Req.: REQ-B001-23. DoD: CA-B001-12.
- [ ] **T-B001-13 · README**
  - Arquitectura real, cómo ejecutar, variables y enlace a `specs/`.
  - Hallazgos: AUD-B-44.

## Cierre
- [ ] CA-B001-01…12 verificados.
- [ ] Hallazgos resueltos: AUD-B-07 (aislamiento), 09, 10 (protegido), 12, 13, 14, 19, 20, 24…28,
  30…45.
