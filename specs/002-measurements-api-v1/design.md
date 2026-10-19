# B-002 — API de mediciones v1 · Diseño

> Estado: 📝 Pendiente de aprobación · Requisitos: [requirements.md](requirements.md)

## 1. Componentes

```
api/v1/measurements.py ──► application/measurements/
                              CreateMeasurement, ListMeasurements, DeleteMeasurement,
                              GenerateMeasurementsReport
                                 │
                                 ├─► domain/measurements/
                                 │     Measurement, Reading, RecognitionInfo, MeasurementSource,
                                 │     ReadingRules (validador §2–3 de reading-rules), PageCursor
                                 │     ports: MeasurementRepository, ReportRenderer, Clock
                                 │
                                 └─► infrastructure/
                                       mongo/measurement_repository.py (MongoMeasurementRepository)
                                       mongo/migrations/m001_measurements.py
                                       reports/reportlab_renderer.py (ReportLabReportRenderer)
api/legacy/measurements_legacy.py ──► mismos casos de uso + LegacyMeasurementMapper
```

## 2. Modelo y persistencia

```python
@dataclass(frozen=True)
class Reading:          # se valida en el constructor con ReadingRules
    systolic: int
    diastolic: int
    pulse: int

@dataclass(frozen=True)
class RecognitionInfo:
    confidence: float
    model_version: str
    edited_fields: frozenset[Literal["systolic", "diastolic", "pulse"]]

@dataclass(frozen=True)
class Measurement:
    id: str | None
    user_id: str
    client_id: UUID
    reading: Reading
    measured_at: datetime          # aware, UTC
    created_at: datetime | None
    source: Literal["camera", "manual"]
    recognition: RecognitionInfo | None
```

Colección `measurements`:
```json
{ "_id": ObjectId, "user_id": ObjectId, "client_id": "uuid-string",
  "systolic": 128, "diastolic": 82, "pulse": 67,
  "measured_at": ISODate(UTC), "created_at": ISODate(UTC),
  "source": "camera",
  "recognition": { "confidence": 0.93, "model_version": "digits-1.0.0", "edited_fields": ["pulse"] },
  "payload_hash": "sha256 del payload canónico",
  "legacy_id": ObjectId | null }
```
Índices:
- `{user_id: 1, client_id: 1}` **único**;
- `{user_id: 1, measured_at: -1, _id: -1}` (listado y paginación).

## 3. Idempotencia (REQ-B002-02/03)

### ADR-B002-01 — Hash del payload canónico + índice único
- **Payload canónico**: JSON con claves ordenadas de `{client_id, systolic, diastolic, pulse,
  measured_at (UTC ISO con segundos), source, recognition (edited_fields ordenado, confidence
  redondeada a 4 decimales)}` → `sha256`.
- **Algoritmo**:
  1. `insert_one(doc)`;
  2. si se lanza `DuplicateKeyError` → `find_one({user_id, client_id})`;
  3. si `payload_hash` coincide → `(existing, created=False)` → 200; si no → `ClientIdConflict` → 409.
- Resuelve la carrera sin transacciones: el índice único decide quién gana y el perdedor compara
  hashes.
- **Alternativa descartada**: hacer primero `find_one` y luego `insert`, porque tiene una
  ventana de carrera que produciría un 500 o un 409 espurio.

## 4. Paginación (REQ-B002-06)

- **Cursor** = `base64url(json({"m": measured_at_iso, "i": str(_id)}))`.
- **Consulta siguiente**:
  ```
  {user_id, $or: [{measured_at: {$lt: m}}, {measured_at: m, _id: {$lt: i}}]}
  ```
  ordenada por `(measured_at -1, _id -1)` con `limit + 1` (para saber si hay más páginas).
- Un cursor inválido → 422 `VALIDATION_ERROR` (`field: cursor`).

## 5. Convivencia legacy (REQ-B002-12)

`LegacyMeasurementMapper` convierte `Measurement` en el formato antiguo:
```json
{ "measurement_id": "<id>", "user_id": "<uid>", "filename": "",
  "result": { "high_pressure": "128", "low_pressure": "82", "pulse": "67", "confidence": 0.93 },
  "timestamp": "<measured_at ISO sin zona, en UTC>" }
```
(`confidence` = `recognition.confidence` o `1.0` si es manual). `/ocr/display-recognize`, si está
activo, crea una `Measurement` con `client_id = uuid4()`, `source = camera` y `model_version =
"legacy-keras"`, **saltándose la validación de rangos**. Ese comportamiento es el actual y la ruta
está deprecada. Se documenta como excepción y lo cubre un test.

## 6. Informe PDF

### ADR-B002-02 — ReportLab (y no xhtml2pdf + Plotly)
- **Problema actual**: el HTML con Plotly (JS) no se renderiza en xhtml2pdf, así que el PDF sale
  sin gráfico. Además hay riesgo de inyección (AUD-B-21) y cinco librerías de PDF instaladas.
- **Decisión**: ReportLab (Python puro, sin dependencias de sistema). Tabla con `platypus` y
  gráfico de líneas con `reportlab.graphics.charts.lineplots`. Los textos se insertan como
  `Paragraph` con escapado XML (`xml.sax.saxutils.escape`).
- **Alternativas descartadas**: WeasyPrint (necesita Pango y Cairo en la imagen, mucho más peso) y
  Matplotlib → PNG (dependencia pesada solo para un gráfico).

### ADR-B002-03 — La zona horaria la decide el cliente
- El servidor guarda UTC. El parámetro `tz` (IANA, `zoneinfo`) controla el formato. Por defecto es
  `UTC`; la app envía la del dispositivo.

### ADR-B002-04 — Se elimina el informe HTML de v1
- La app pinta un gráfico nativo (App A-003). Un HTML con JS en un WebView es una superficie de
  ataque sin valor añadido.

## 7. Migración (REQ-B002-11)

```
uv run python -m app.infrastructure.mongo.migrations.m001_measurements --dry-run [--legacy-tz Europe/Madrid]
```
1. Recorre `recognition_results` ordenado por `_id`.
2. Por documento: intenta construir `Reading(int(hp), int(lp), int(p))`. Si falla →
   `skipped.append({legacy_id, reason})`.
3. Con `client_id = uuid5(NAMESPACE_TENSOSCAN, str(_id))`, hace
   `update_one({user_id, client_id}, {$setOnInsert: doc}, upsert=True)`. Por eso es idempotente.
4. Al final escribe el informe en el log y en `migration_reports` (`{migration: "m001", ran_at,
   migrated, skipped:[...]}`).

**Cambios de hora**: se usa `zoneinfo.ZoneInfo("Europe/Madrid")`. Un `timestamp` naive en la hora
repetida de octubre se interpreta con `fold=0` (primera ocurrencia) y uno en la hora inexistente de
marzo se normaliza sumando 1 h. Ambos casos se cuentan en el informe de migración
(`ambiguous_local_times`, `nonexistent_local_times`) y se cubren con tests.

`NAMESPACE_TENSOSCAN` es un UUID fijo declarado en el código. **Nunca cambia**, porque la app
reconcilia por `client_id`.

## 8. Retirada del legacy (REQ-B002-13)

PR dedicado (T-B002-09):
- borrar `infrastructure/legacy_ocr/`, `api/legacy/ocr_legacy.py`, el extra `legacy-ocr`,
  `ia_models/` y las rutas de `api/legacy/measurements_legacy.py`;
- borrar los alias de auth y chatbot cuando lo permita A-003;
- **no** borrar la colección `recognition_results`. Se archiva (export) y se borra manualmente
  pasados 90 días.

## 9. Testing

| Nivel | Casos |
|-------|-------|
| Unit (domain) | `ReadingRules` con **todos** los `validation_cases` (`parametrize`), fechas límite, coherencia `source`/`recognition`, payload canónico (estable ante el orden de claves y ante `edited_fields` desordenado), codificación y decodificación del cursor |
| Unit (application) | `CreateMeasurement`: nuevo / reintento igual / conflicto (repositorio *fake*) |
| Integración | POST 201/200/409/422; carrera con `asyncio.gather` de 2 POST → {201, 200}; paginación de 120 documentos; aislamiento entre usuarios; DELETE 204/404; PDF (cabeceras, `%PDF`, texto extraído con `pypdf` que contiene las horas locales); `tz` inválida → 422 |
| Migración | Colección legacy sembrada con valores válidos, `"12O"`, `""` y `"300"` → informe correcto; segunda ejecución sin duplicados |
| Legacy | Las rutas antiguas devuelven el formato antiguo y ven los datos v1 |
| Contrato | Paths, códigos y forma de los JSON frente a `api-v1.md` §3 |
