# B-002 — API de mediciones v1 · Requisitos

> Estado: 📝 Pendiente de aprobación · Tipo: Feature + migración de datos
> Depende de: B-001 · Contratos: `api-v1.md` §3 y §6, `reading-rules.md`
> Resuelve: AUD-B-01…08 (retirada), 10, 11, 15…18, 21…23 · Habilita: App A-002 fase D y A-003

## 1. Contexto

Con el reconocimiento en el dispositivo (App ADR-A002-01), el servidor recibe **mediciones ya
confirmadas** por el usuario. Debe validarlas con las mismas reglas que la app, aceptar
reintentos sin duplicar (la app sincroniza offline-first), servir el historial paginado y generar
el informe PDF. Hay datos legacy en `recognition_results` (valores como texto, fechas naive) que
deben migrarse sin pérdida.

## 2. Historias

- **HU-B002-1**: Como app, quiero enviar una medición confirmada y que un reintento no cree un
  duplicado.
- **HU-B002-2**: Como app, quiero recibir el historial paginado y ordenado.
- **HU-B002-3**: Como usuario, quiero descargar un PDF con mis mediciones, un gráfico y mis
  fechas en mi zona horaria.
- **HU-B002-4**: Como usuario existente, quiero seguir viendo mis mediciones antiguas.

## 3. Requisitos (EARS)

### Creación
- **REQ-B002-01**: `POST /v1/measurements` DEBERÁ validar el cuerpo con `reading-rules.md` §2–3.1
  (rangos, SYS > DIA, `measured_at` ni futuro > 5 min ni anterior a 2000), además de `source`
  frente a `recognition` y los límites de `recognition` (contrato §3.1). SI falla, ENTONCES
  responderá 422 `MEASUREMENT_INVALID` con los `details` de cada fallo.
- **REQ-B002-02**: La creación DEBERÁ ser idempotente por `(user_id, client_id)`:
  - si no existe → 201;
  - si existe con el **mismo payload canónico** → 200 con el recurso existente;
  - si existe con un payload distinto → 409 `CLIENT_ID_CONFLICT`.
- **REQ-B002-03**: La unicidad DEBERÁ garantizarla un índice único `(user_id, client_id)`. Una
  carrera entre dos peticiones iguales DEBERÁ resolverse como 201 + 200 (nunca 500 ni 409).
- **REQ-B002-04**: `measured_at` DEBERÁ guardarse en UTC. `created_at` lo asigna el servidor.
- **REQ-B002-05**: Los avisos de `reading-rules.md` §3.2 NO DEBERÁN impedir la creación (el
  usuario ya los confirmó en la app).

### Consulta y borrado
- **REQ-B002-06**: `GET /v1/measurements` DEBERÁ devolver solo las del usuario autenticado,
  ordenadas por `(measured_at desc, _id desc)`, con paginación por cursor opaco (base64url de
  `measured_at|id`), `limit` entre 1 y 200 (por defecto 50) y filtros `from` y `to`.
- **REQ-B002-07**: `DELETE /v1/measurements/{id}` DEBERÁ borrar solo si pertenece al usuario y
  responder 404 en caso contrario (sin revelar si existe).

### Informe
- **REQ-B002-08**: `GET /v1/measurements/report?format=pdf&from&to&tz` DEBERÁ generar un PDF con:
  - cabecera (nombre del usuario, rango y fecha de generación en `tz`);
  - resumen (número de lecturas y media, mínimo y máximo de SYS, DIA y PUL);
  - gráfico de evolución;
  - tabla (fecha y hora locales, SYS, DIA y PUL);
  - pie con el aviso de que el informe es orientativo y no un diagnóstico.

  Todo el texto interpolado DEBERÁ escaparse.
- **REQ-B002-09**: SI no hay mediciones en el rango, ENTONCES responderá 404. SI `tz` no es una
  zona IANA válida, ENTONCES responderá 422.
- **REQ-B002-10**: El endpoint HTML del informe DEBERÁ eliminarse de v1. La ruta legacy
  `/ocr/user/measurements/html` se mantiene hasta su retirada, generada con plantillas con
  *autoescape*.

### Migración y convivencia
- **REQ-B002-11**: DEBERÁ existir un script idempotente
  `python -m app.infrastructure.mongo.migrations.m001_measurements` (con `--dry-run`) que copie
  `recognition_results` → `measurements`:
  - parsea los valores a `int`;
  - `client_id = uuid5(NAMESPACE_TENSOSCAN, str(legacy_id))`;
  - `measured_at = created_at = timestamp` interpretado en `--legacy-tz` (por defecto
    `Europe/Madrid`, decisión del responsable) y convertido a UTC. Las horas ambiguas del cambio de
    hora de otoño se resuelven con `fold=0` y las inexistentes de primavera se desplazan 1 h
    hacia delante; ambas se anotan en el informe;
  - `source = camera` y `recognition = {confidence, model_version: "legacy-keras", edited_fields: []}`.

  Los documentos inválidos (no numéricos o fuera de rango) **no** se migran y se listan en un
  informe. La colección legacy **no** se modifica.
- **REQ-B002-12**: Durante la convivencia, las rutas legacy de mediciones (`/ocr/measurements`,
  `/ocr/remove/measurements/{id}`, `/ocr/user/measurements/pdf|html`) DEBERÁN leer y escribir en
  `measurements`, adaptando la respuesta al formato antiguo (valores como `string` y `filename`
  vacío). `POST /ocr/display-recognize` (si `ENABLE_LEGACY_OCR`) DEBERÁ escribir también en
  `measurements`.
- **REQ-B002-13**: CUANDO la app A-002 esté publicada y adoptada (criterio: menos del 5 % de
  peticiones a rutas legacy de mediciones durante 14 días, o decisión explícita del responsable),
  el OCR legacy, TensorFlow, OpenCV, `ia_models/` y las rutas `/ocr/*` DEBERÁN eliminarse.

### Observabilidad mínima
- **REQ-B002-14**: Cada petición a una ruta legacy DEBERÁ registrarse (ruta y `user_id`) para
  medir el criterio de REQ-B002-13.

## 4. Criterios de aceptación

| ID | Escenario | Req. |
|----|-----------|------|
| CA-B002-01 | Dado un payload válido, cuando hago POST, entonces → 201 con `id` y `created_at` | 01 |
| CA-B002-02 | Dado el mismo payload y el mismo `client_id`, cuando repito el POST, entonces → 200 con el mismo `id`, y existe un único documento | 02 |
| CA-B002-03 | Dado el mismo `client_id` con `pulse` distinto, cuando hago POST, entonces → 409 `CLIENT_ID_CONFLICT` | 02 |
| CA-B002-04 | Dados dos POST idénticos concurrentes, cuando se procesan, entonces → {201, 200} | 03 |
| CA-B002-05 | Dados todos los `validation_cases` de los casos dorados, cuando se envían, entonces los errores coinciden en `field` y `code` (y los avisos no bloquean) | 01, 05 |
| CA-B002-06 | Dadas 120 mediciones, cuando pagino con `limit=50`, entonces obtengo 50 + 50 + 20, sin repeticiones y en orden | 06 |
| CA-B002-07 | Dada una medición de otro usuario, cuando hago GET o DELETE por su id, entonces → 404 | 07 |
| CA-B002-08 | Dado `tz=Europe/Madrid`, cuando genero el PDF, entonces las horas salen en hora local y el PDF contiene el gráfico y la tabla | 08 |
| CA-B002-09 | Dado un valor legacy `"12O"`, cuando ejecuto la migración, entonces no se migra y aparece en el informe; si la ejecuto dos veces, no se crean duplicados | 11 |
| CA-B002-10 | Dada la app de `master`, cuando lista y borra mediciones por las rutas legacy, entonces ve también las creadas por la v1 | 12 |

## 5. Fuera de alcance
- Edición de una medición ya guardada (se borra y se crea de nuevo). Puede añadirse en v1.1
  como `PATCH` con control de concurrencia.
- Exportar a CSV.
- Compartir el informe con terceros por enlace.

## 6. Preguntas abiertas

| ID | Pregunta | Propuesta |
|----|----------|-----------|
| ~~Q-B002-01~~ | ✅ **Resuelta**: los `timestamp` legacy se interpretan como `Europe/Madrid` | — |
| Q-B002-02 | ¿Hay datos de producción reales que migrar o solo de desarrollo? | Determina si se necesita una ventana de migración |
