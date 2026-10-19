# B-002 — API de mediciones v1 · Tareas

> Requisito previo: B-001 (al menos T-B001-01…07). Commit: `T-B002-xx | <descripción en inglés>`.

- [x] **T-B002-01 · Dominio de mediciones + `ReadingRules`**
  - Fixture de los casos dorados leída desde `specs/contracts/reading-golden-cases.json` (sin
    copias). Payload canónico.
  - Req.: REQ-B002-01, 05 · Hallazgos: AUD-B-15.
- [x] **T-B002-02 · `MongoMeasurementRepository` + índices**
  - Inserción idempotente (ADR-B002-01), listado con cursor y borrado con propietario.
  - Incluye `delete_all_for_user` (lo usa T-B001-14).
  - Req.: REQ-B002-02…04, 06, 07 · Hallazgos: AUD-B-16, 17.
- [x] **T-B002-03 · Casos de uso `CreateMeasurement`, `ListMeasurements` y `DeleteMeasurement`**
  - Req.: REQ-B002-01…07.
- [x] **T-B002-04 · Router `/v1/measurements` + schemas + tests de integración y de contrato**
  - Hallazgos: AUD-B-11, 18 · DoD: CA-B002-01…07.
- [x] **T-B002-05 · `GenerateMeasurementsReport` + `ReportLabReportRenderer` + endpoint `/report`**
  - Eliminar `html_render.py` (salvo la plantilla con *autoescape* para el alias legacy),
    `pdf_generator.py` y xhtml2pdf.
  - Req.: REQ-B002-08…10 · Hallazgos: AUD-B-21, 22, 23 · DoD: CA-B002-08.
- [x] **T-B002-06 · Script de migración m001 + tests** (incluye los casos de cambio de hora)
  - Req.: REQ-B002-11 · DoD: CA-B002-09.
- [ ] **T-B002-07 · Rutas legacy de mediciones sobre la nueva colección + log de uso**
  - Req.: REQ-B002-12, 14 · DoD: CA-B002-10.
- [ ] **T-B002-08 · Copiar a la app el contrato `api-v1.md` (versión final) y avisar a A-002 fase D**
  - Incluye ejemplos de petición y respuesta reales generados por los tests (fixtures JSON en
    `specs/contracts/examples/`).
- [ ] **T-B002-09 · (Diferida, REQ-B002-13) Retirar el OCR legacy, TensorFlow y las rutas `/ocr/*`**
  - Hallazgos: AUD-B-01…08, 10 (cierre).

## Cierre
- [ ] CA-B002-01…10 verificados.
- [ ] Migración ejecutada en cada entorno con informe archivado.
