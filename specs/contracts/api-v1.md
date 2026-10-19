# Contrato API REST v1 — TensoScan

> **Fuente canónica**: `TensoScan_Backend/specs/contracts/api-v1.md`.
> Copia sincronizada en `TensoScan_App/specs/contracts/api-v1.md`. Cualquier cambio se hace
> **primero aquí**, sube la versión del documento y se replica en la app dentro del mismo PR o en
> uno enlazado.
>
> Versión del contrato: **1.2.0** · Estado: 📝 Pendiente de aprobación

## 1. Convenciones generales

| Aspecto | Regla |
|---------|-------|
| Base path | `/v1`. Las rutas legacy (sección 6) conviven hasta su retirada |
| Formato | `application/json; charset=utf-8`, con campos en `snake_case` |
| Autenticación | `Authorization: Bearer <access_token>` (JWT HS256) en todo salvo `register`, `login`, `refresh` y `health` |
| Fechas | ISO 8601 **con zona horaria**, por ejemplo `2026-10-19T08:15:00+02:00`. El servidor guarda y devuelve en UTC (`Z`) |
| Identificadores | `id` es un ObjectId de Mongo serializado como string. `client_id` es un UUID v4 generado por la app |
| Paginación | Por cursor: `?limit=50&cursor=<opaque>` → `{ "items": [...], "next_cursor": "..." \| null }` |
| Idempotencia | `POST /v1/measurements` es idempotente por `(user, client_id)` |
| Versionado | Un cambio incompatible exige `/v2`. Añadir campos opcionales a una respuesta es compatible; la app ignora los campos desconocidos |
| Tamaño máximo de petición | 64 KB (no hay endpoints de subida de imágenes en v1) |

### 1.1 Sobre de error (todas las respuestas 4xx y 5xx)

```json
{
  "error": {
    "code": "MEASUREMENT_INVALID",
    "message": "Human readable message (en)",
    "details": [ { "field": "systolic", "code": "OUT_OF_RANGE" } ]
  }
}
```

| HTTP | `error.code` | Cuándo |
|------|--------------|--------|
| 400 | `BAD_REQUEST` | JSON mal formado |
| 401 | `UNAUTHORIZED` | Falta el token o es inválido |
| 401 | `TOKEN_EXPIRED` | El access token ha caducado → la app debe llamar a `/v1/auth/refresh` |
| 401 | `INVALID_CREDENTIALS` | Login fallido |
| 403 | `FORBIDDEN` | El recurso pertenece a otro usuario |
| 404 | `NOT_FOUND` | El recurso no existe |
| 409 | `EMAIL_ALREADY_REGISTERED` | Registro con un email existente |
| 409 | `CLIENT_ID_CONFLICT` | Mismo `client_id` con un payload distinto |
| 422 | `VALIDATION_ERROR` / `MEASUREMENT_INVALID` | Falla la validación (`details` indica qué campo) |
| 429 | `RATE_LIMITED` | Demasiadas peticiones (login y chatbot) |
| 500 | `INTERNAL_ERROR` | Error no controlado (sin trazas en la respuesta) |
| 503 | `UPSTREAM_UNAVAILABLE` | Proveedor externo caído (chatbot) |

Los códigos de `details[].code` de medición son los definidos en `reading-rules.md` §3
(`REQUIRED`, `OUT_OF_RANGE`, `SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC`, `IN_FUTURE`...) más los de
coherencia de `recognition` de la sección 3.2.

Ejemplos reales de petición y respuesta (generados por los tests del backend y comprobados en
cada ejecución del CI): [`examples/`](examples/).

## 2. Autenticación

### `POST /v1/auth/register`
```json
// request
{ "name": "Ana", "last_name": "García", "email": "ana@example.com", "password": "********" }
// 201
{ "id": "6712...", "email": "ana@example.com", "name": "Ana", "last_name": "García" }
```
- La contraseña debe tener entre 8 y 128 caracteres. El email se normaliza a minúsculas.
- Errores: `409 EMAIL_ALREADY_REGISTERED` y `422 VALIDATION_ERROR`.

### `POST /v1/auth/login` (`application/x-www-form-urlencoded`, compatible con OAuth2 password)
```
username=ana@example.com&password=********
```
```json
// 200
{ "access_token": "eyJ...", "refresh_token": "rt_...", "token_type": "bearer", "expires_in": 1800 }
```
- Errores: `401 INVALID_CREDENTIALS` (el mensaje es el mismo exista o no el email) y
  `429 RATE_LIMITED`.

### `POST /v1/auth/refresh`
```json
// request
{ "refresh_token": "rt_..." }
// 200 → mismo cuerpo que login (el refresh token ROTA: el anterior queda revocado)
```
- Errores: `401 UNAUTHORIZED` (refresh inválido, caducado o reutilizado → se revoca toda la
  familia de tokens).

### `POST /v1/auth/logout`
```json
{ "refresh_token": "rt_..." }   // 204
```

### `GET /v1/users/me`
```json
// 200
{ "id": "6712...", "email": "ana@example.com", "name": "Ana", "last_name": "García" }
```

### `DELETE /v1/users/me` — supresión de cuenta (RGPD art. 17 y requisito de Google Play)
```json
// request
{ "password": "********" }
// 204 No Content
```
- Borra de forma definitiva al usuario, **todas** sus mediciones y todos sus refresh tokens.
- `401 INVALID_CREDENTIALS` si la contraseña no coincide (evita que un token robado borre la cuenta).
- Es idempotente solo en el sentido de que, tras el 204, cualquier token del usuario devuelve
  `401 UNAUTHORIZED`.

## 3. Mediciones

### 3.1 Recurso `Measurement`

```json
{
  "id": "6713a1...",
  "client_id": "5b0b6f0e-6a1f-4d8e-9d0a-2f1c3b7e9a10",
  "systolic": 128,
  "diastolic": 82,
  "pulse": 67,
  "measured_at": "2026-10-19T06:15:00Z",
  "created_at": "2026-10-19T06:15:42Z",
  "source": "camera",
  "recognition": {
    "confidence": 0.93,
    "model_version": "digits-1.0.0",
    "edited_fields": ["pulse"]
  }
}
```

| Campo | Tipo | Reglas |
|-------|------|--------|
| `client_id` | UUID v4 | Obligatorio y único por usuario |
| `systolic` | int | `reading-rules.md` §2 |
| `diastolic` | int | `reading-rules.md` §2 |
| `pulse` | int | `reading-rules.md` §2 |
| `measured_at` | datetime con zona | No puede estar a más de 5 minutos en el futuro respecto al reloj del servidor ni ser anterior a `2000-01-01`. Se guarda en UTC con precisión de segundos (se descartan las fracciones) |
| `source` | enum | `camera` (propuesto por reconocimiento) · `manual` (introducido a mano) |
| `recognition` | objeto \| null | Obligatorio si `source=camera` y prohibido si `source=manual` |
| `recognition.confidence` | float | `[0, 1]` |
| `recognition.model_version` | string | ≤ 64 caracteres (`mlkit-text-v2` o `digits-x.y.z`) |
| `recognition.edited_fields` | array | Subconjunto de `systolic`, `diastolic` y `pulse`; indica qué valores corrigió el usuario |
| `created_at` | datetime | Solo de servidor |

### `POST /v1/measurements`
- Cuerpo: el recurso sin `id` ni `created_at`.
- `201 Created` y el recurso, si es nuevo.
- `200 OK` y el recurso existente, si ya existe `(user, client_id)` con **el mismo** payload
  (reintento idempotente). El payload se compara en forma canónica: el mismo instante en otro
  desfase horario, las fracciones de segundo, el orden de `edited_fields` y la `confidence`
  redondeada a 4 decimales no cuentan como diferencias.
- `409 CLIENT_ID_CONFLICT` si existe con un payload distinto.
- `422 MEASUREMENT_INVALID` con `details` si falla `reading-rules.md` §3.1 o la coherencia de
  `source` y `recognition` (sección 3.2). Un valor ausente o `null` en `systolic`, `diastolic` o
  `pulse` se informa como `REQUIRED`.
- `422 VALIDATION_ERROR` si el cuerpo no tiene la forma esperada (`client_id` que no es UUID v4,
  `measured_at` sin zona horaria, valores no enteros, `source` desconocido...). `details[].field` es
  la ruta del campo y `details[].code` el tipo de error de validación.
- Ejemplos: `examples/measurements-create.201.json`, `measurements-create-replay.200.json`,
  `measurements-create-conflict.409.json` y `measurements-create-invalid.422.json`.

### 3.2 Códigos de `details[].code` de una medición

Además de los de `reading-rules.md` §3.1 (`REQUIRED`, `OUT_OF_RANGE`,
`SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC`, `IN_FUTURE`, `TOO_OLD`):

| `field` | `code` | Cuándo |
|---------|--------|--------|
| `recognition` | `REQUIRED` | `source=camera` sin `recognition` |
| `recognition` | `NOT_ALLOWED` | `source=manual` con `recognition` |
| `recognition.confidence` | `OUT_OF_RANGE` | Fuera de `[0, 1]` |
| `recognition.model_version` | `REQUIRED` | Cadena vacía |
| `recognition.model_version` | `TOO_LONG` | Más de 64 caracteres |

Los avisos de `reading-rules.md` §3.2 **nunca** bloquean la creación.

### `GET /v1/measurements`
- Query: `from` y `to` (datetime, opcionales, ambos inclusivos, filtran por `measured_at`; sin zona
  se interpretan como UTC), `limit` (1–200, por defecto 50) y `cursor`.
- Orden: `measured_at` descendente y, en caso de empate, `id` descendente.
- `200 { "items": [Measurement], "next_cursor": "..." | null }`. `next_cursor` es opaco: se pasa tal
  cual en la siguiente petición y es `null` en la última página.
- `422 VALIDATION_ERROR` con `{ "field": "cursor", "code": "INVALID" }` si el cursor no lo emitió
  esta API, y `422 VALIDATION_ERROR` si `limit` o las fechas están fuera de rango o mal formadas.
- Ejemplo: `examples/measurements-list-page.200.json`.

### `DELETE /v1/measurements/{id}`
- `204`; `404 NOT_FOUND` si no existe, es de otro usuario o el `id` está mal formado (no se revela
  su existencia). Ejemplo del sobre de error: `examples/error-envelope.404.json`.

### `GET /v1/measurements/report?format=pdf`
- Query: `format` (opcional; el único valor admitido es `pdf`), `from`, `to` (opcionales, como en
  el listado) y `tz` (zona IANA para formatear las fechas, por ejemplo `Europe/Madrid`; por
  defecto, la zona configurada en el servidor `TENSOSCAN_DEFAULT_REPORT_TZ`, que vale
  `Europe/Madrid`).
- `200 application/pdf` con `Content-Disposition: attachment; filename="tensoscan-report-YYYYMMDD.pdf"`
  (`YYYYMMDD` es la fecha de generación en la zona `tz`). El PDF, en español, lleva cabecera
  (paciente, periodo y fecha de generación), resumen (número de lecturas y media, mínimo y máximo
  de SYS, DIA y PUL), gráfico de evolución, tabla con las fechas y horas locales y, en cada página,
  un pie que avisa de que el informe es orientativo y no un diagnóstico.
- `404 NOT_FOUND` si no hay mediciones en el rango.
- `422 VALIDATION_ERROR` con `{ "field": "tz", "code": "INVALID_TIMEZONE" }` si `tz` no es una
  zona IANA, y `422 VALIDATION_ERROR` sobre `format` para cualquier valor distinto de `pdf`.
- Nota: el formato `html` se **elimina** de v1. La app pinta su propio gráfico nativo (ver
  ADR-B002-04).

## 4. Chatbot (sin cambios funcionales, solo se traslada)

### `POST /v1/assistant/messages`
```json
// request
{ "prompt": "¿Qué significa 135/85?", "history": [ { "role": "user", "content": "..." }, { "role": "assistant", "content": "..." } ] }
// 200
{ "response": "..." }
```
- `history` admite como máximo 20 mensajes de 2.000 caracteres cada uno. `prompt` admite como
  máximo 2.000 caracteres.
- Errores: `429 RATE_LIMITED` y `503 UPSTREAM_UNAVAILABLE`.

## 5. Operación

### `GET /health` → `200 { "status": "ok", "version": "x.y.z" }` (sin autenticación y sin prefijo `/v1`)

## 6. Rutas legacy y calendario de retirada

| Legacy | Sustituta v1 | Retirada |
|--------|--------------|----------|
| `POST /register` | `POST /v1/auth/register` | Cuando la app A-003 esté publicada |
| `POST /login` | `POST /v1/auth/login` | Ídem |
| `GET /me` | `GET /v1/users/me` | Ídem |
| `POST /ai/chatbot` | `POST /v1/assistant/messages` | Ídem |
| `GET /ocr/measurements` | `GET /v1/measurements` | Cuando la app A-002 esté publicada |
| `DELETE /ocr/remove/measurements/{id}` | `DELETE /v1/measurements/{id}` | Ídem |
| `GET /ocr/user/measurements/pdf` | `GET /v1/measurements/report?format=pdf` | Ídem |
| `GET /ocr/user/measurements/html` | — (gráfico nativo en la app) | Ídem |
| `POST /ocr/display-recognize`, `POST /ocr/recognize` | — (el reconocimiento pasa al dispositivo) | Ídem; deja de cargarse TensorFlow en el servidor |

Durante la convivencia, las rutas legacy leen y escriben sobre la **nueva** colección
`measurements` mediante un adaptador (ver B-002 design §5), de modo que una app antigua y una
nueva ven los mismos datos.

## 7. Historial de cambios del contrato

| Versión | Fecha | Cambio |
|---------|-------|--------|
| 1.0.0 | 2026-10-19 | Versión inicial |
| 1.1.0 | 2026-10-19 | Añade `DELETE /v1/users/me`; la zona horaria por defecto del informe pasa a ser `Europe/Madrid` |
| 1.2.0 | 2026-10-19 | Precisa las mediciones al implementar B-002: códigos de coherencia de `recognition` (§3.2), comparación canónica en la idempotencia, `measured_at` con precisión de segundos, cursor inválido y `tz` inválida (422), `format` opcional (solo `pdf`), `404` para ids mal formados, contenido del PDF y ejemplos reales en `examples/`. No cambia ningún endpoint ni campo existente |
