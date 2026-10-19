# B-003 — Modelo de detección de dígitos · Tareas

> Puede empezar en paralelo a B-001. Commit: `T-B003-xx | <descripción en inglés>`.
> Las tareas marcadas con 🧑 necesitan una acción del usuario (datos, GPU o decisión).

## Fase 1 — Base y reglas
- [x] **T-B003-01 · Andamiaje de `ml/`**
  - `pyproject` propio, CLI vacía con subcomandos, configs, ruff, mypy, pytest y job de CI
    `ml-tests`.
  - Req.: REQ-B003-01…03, 22.
  - Nota de implementación: la CLI es `argparse` (sin dependencias); cada subcomando vive en `tensoscan_ml/commands/`.
    El CI tiene dos jobs: `ml-tests` (núcleo ligero, lint, auditoría de licencias y tests rápidos) y `ml-tests-heavy`
    (todos los extras, mypy y todos los tests en CPU). `release` queda como esqueleto (T-B003-13).
- [ ] **T-B003-02 · `tensoscan_ml.reading`: parser y validador oficiales**
  - Tests con **todos** los casos dorados (leídos de `specs/contracts/`).
  - Req.: REQ-B003-04 · DoD: CA-B003-03.

## Fase 2 — Datos sintéticos
- [ ] **T-B003-03 · Render de glifos de 7 segmentos**
  - Tests geométricos: segmentos por clase y caja frente a máscara con IoU ≥ 0,9.
  - Req.: REQ-B003-05 · DoD: CA-B003-02.
- [ ] **T-B003-04 · Plantillas de layout + elementos auxiliares + escritor YOLO y metadatos**
  - Test: el parser oficial aplicado sobre las etiquetas exactas de cada imagen sintética de una
    plantilla cubierta reproduce los valores renderizados. Así el generador y el parser son
    coherentes entre sí.
  - Req.: REQ-B003-06, 08.
- [ ] **T-B003-05 · Escena y aumentos + CLI `synth` con semilla**
  - Req.: REQ-B003-07, 08 · DoD: CA-B003-01.
  - Entregable: una galería de 50 muestras (`docs/synth-samples.png`) para revisarla con el
    usuario.

## Fase 3 — Datos reales
- [ ] **T-B003-06 · Ingesta (`naming`, `ingest`, `split`) + documento del protocolo**
  - Tests con vídeos sintéticos generados en el propio test.
  - Req.: REQ-B003-09…12 · DoD: CA-B003-04.
- [ ] 🧑 **T-B003-06b · Aportar datos reales (etapa 2)**
  - Hoy: fotos de 1 tensiómetro con varios valores (`labels.csv`). Para `production`: ≥ 3 dispositivos.
  - Bloquea T-B003-09 (*fine-tuning*) y la evaluación de aceptación, no el resto del código.

## Fase 4 — Modelo
- [ ] **T-B003-07 · Harness de evaluación (`eval`)**
  - Fuentes: *checkpoint*, `tflite` y `detections.jsonl`.
  - Tests con detecciones sintéticas perfectas → 100 %.
  - Req.: REQ-B003-16, 17.
- [ ] 🧑 **T-B003-08 · *Spike* de arquitectura (ADR-B003-02)**
  - Entrenar las 2 candidatas con el mismo presupuesto sobre sintético. Exportar y `verify`.
    Registrar el resultado y la decisión en el ADR.
  - Requiere GPU (Q-B003-03).
  - Req.: REQ-B003-14.
- [ ] **T-B003-09 · Pipeline de entrenamiento de 2 etapas**
  - Configs, semillas, *checkpoints* y curvas.
  - Req.: REQ-B003-15.
- [ ] **T-B003-10b · Modelo «dummy» conforme al contrato (pesos aleatorios)**
  - `tensoscan-ml dummy-model` genera un `.tflite` + sidecar con la E/S exacta del contrato, sin valor
    predictivo. Permite completar y testear en la app todo el código de `LiteRtDigitDetector`
    (carga, verificación sha256, preprocesado, decodificación y latencia) **antes** de entrenar.
  - Req.: REQ-B003-18, 19.
- [ ] **T-B003-10 · Exportación al contrato** (`OutputContractWrapper`, float16 e int8)
  - Req.: REQ-B003-18.
- [ ] **T-B003-11 · `verify`** (E/S, operaciones *builtin*, clases, tamaño, latencia y set dorado)
  - Req.: REQ-B003-19 · DoD: CA-B003-05.
- [ ] **T-B003-12 · `prelabel` + guía de anotación en CVAT** (subconjunto de ~300 frames reales)
  - Req.: REQ-B003-13.

## Fase 5 — Release
- [ ] **T-B003-13 · `release` v1.0.0**
  - Umbrales, sidecar, *model card*, GitHub Release y PR a la app (App T-A002-33/35).
  - Canales `beta` y `production` (requirements §1). Con un solo dispositivo solo es posible `beta`.
  - Req.: REQ-B003-20, 21 · DoD: CA-B003-06, 07 y las métricas de requirements §1 en real.

## Cierre
- [ ] `report.md` de la release archivado en `ml/reports/digits-1.0.0.md`.
- [ ] Hallazgos resueltos: AUD-B-06.
