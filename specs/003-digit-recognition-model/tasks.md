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
- [x] **T-B003-02 · `tensoscan_ml.reading`: parser y validador oficiales**
  - Tests con **todos** los casos dorados (leídos de `specs/contracts/`).
  - Req.: REQ-B003-04 · DoD: CA-B003-03.
  - Nota de implementación: el validador omite `POSSIBLE_DUPLICATE` (solo app, necesita el historial); `IN_FUTURE` y
    `TOO_OLD` solo se evalúan si se pasa `measured_at`.

## Fase 2 — Datos sintéticos
- [x] **T-B003-03 · Render de glifos de 7 segmentos**
  - Tests geométricos: segmentos por clase y caja frente a máscara con IoU ≥ 0,9.
  - Req.: REQ-B003-05 · DoD: CA-B003-02.
  - Nota de implementación: el margen de la etiqueta es el 2 % por lado de cada dimensión de la caja (4 % en total,
    no «+4 % de H» en absoluto): con un margen absoluto el IoU ≥ 0,9 es imposible para el dígito `1`, que es muy
    estrecho. Los segmentos usan juntas en inglete (el `1` mide lo mismo que el `8`, como en los casos dorados).
- [x] **T-B003-04 · Plantillas de layout + elementos auxiliares + escritor YOLO y metadatos**
  - Test: el parser oficial aplicado sobre las etiquetas exactas de cada imagen sintética de una
    plantilla cubierta reproduce los valores renderizados. Así el generador y el parser son
    coherentes entre sí.
  - Req.: REQ-B003-06, 08.
  - Nota de implementación: el test de coherencia se cumple al 100 % con escenas sin perspectiva, tamaño legible y
    fuentes «compatibles con el parser». Con fuentes ordinarias el parser de la especificación no une un `1` a su
    vecino izquierdo (hallazgo F1 del README); con perspectiva y rotación quedan casos límite (F2). Ambos están
    documentados y probados en `tests/test_synth_layouts.py`; no se ha tocado el contrato.
- [x] **T-B003-05 · Escena y aumentos + CLI `synth` con semilla**
  - Req.: REQ-B003-07, 08 · DoD: CA-B003-01.
  - Entregable: una galería de 50 muestras (`docs/synth-samples.png`) para revisarla con el
    usuario.
  - Nota de implementación: `ml/docs/synth-samples.png` (893 KiB) se genera con
    `tensoscan-ml synth --gallery docs/synth-samples.png`.

## Fase 3 — Datos reales
- [x] **T-B003-06 · Ingesta (`naming`, `ingest`, `split`) + documento del protocolo**
  - Tests con vídeos sintéticos generados en el propio test.
  - Req.: REQ-B003-09…12 · DoD: CA-B003-04.
  - Nota de implementación: admite vídeos y fotos con `labels.csv` (columnas `roi_x,roi_y,roi_w,roi_h` opcionales
    para `--roi-mode manual`). El pHash es propio (DCT, 64 bits) para no depender de `imagehash`/SciPy. El recorte
    `center` sigue `RoiGeometry` de la app (alto = mín(1,25·ancho, 0,60·H)), por lo que solo es 4:5 con imágenes
    verticales. Protocolo en `ml/docs/capture-protocol.md`.
- [ ] 🧑 **T-B003-06b · Aportar datos reales (etapa 2)**
  - Hoy: fotos de 1 tensiómetro con varios valores (`labels.csv`). Para `production`: ≥ 3 dispositivos.
  - Bloquea T-B003-09 (*fine-tuning*) y la evaluación de aceptación, no el resto del código.

## Fase 4 — Modelo
- [x] **T-B003-07 · Harness de evaluación (`eval`)**
  - Fuentes: *checkpoint*, `tflite` y `detections.jsonl`.
  - Tests con detecciones sintéticas perfectas → 100 %.
  - Req.: REQ-B003-16, 17.
  - Nota de implementación: letterbox y decodificador son portes numéricos del Kotlin (`tensoscan_ml/detector/`).
    Se decodifica con umbral 0,05 para el mAP y el parser ve la vista de la app (umbral 0,40, máx. 32). La fuente
    `tflite` se prueba de extremo a extremo en T-B003-10b/11; la fuente `checkpoint` llega con T-B003-09.
- [ ] 🧑 **T-B003-08 · *Spike* de arquitectura (ADR-B003-02)**
  - Entrenar las 2 candidatas con el mismo presupuesto sobre sintético. Exportar y `verify`.
    Registrar el resultado y la decisión en el ADR.
  - Requiere GPU (Q-B003-03).
  - Req.: REQ-B003-14.
- [x] **T-B003-09 · Pipeline de entrenamiento de 2 etapas**
  - Configs, semillas, *checkpoints* y curvas.
  - Req.: REQ-B003-15.
  - Nota de implementación: detector anclaje-libre propio estilo YOLOX-Nano (0,69 M parámetros, `width=0.25`), con
    pérdida SimOTA, EMA, AdamW y coseno; arquitectura enchufable (`ARCHITECTURES`) sin código AGPL. El spike
    (T-B003-08) decide la arquitectura final. La etapa 2 se omite con aviso mientras no haya fotogramas reales con
    cajas (`--require-real` lo convierte en error). Prueba de humo en CPU: `configs/train/smoke.yaml`.
- [x] **T-B003-10b · Modelo «dummy» conforme al contrato (pesos aleatorios)**
  - `tensoscan-ml dummy-model` genera un `.tflite` + sidecar con la E/S exacta del contrato, sin valor
    predictivo. Permite completar y testear en la app todo el código de `LiteRtDigitDetector`
    (carga, verificación sha256, preprocesado, decodificación y latencia) **antes** de entrenar.
  - Req.: REQ-B003-18, 19.
  - Nota de implementación: hay dos modelos (`random`: pesos aleatorios de la arquitectura real; `fixed`: ignora la
    imagen y devuelve siempre la lectura 128/82/67 con un duplicado y una caja de baja puntuación, para probar de
    extremo a extremo decodificador y parser). Ambos se generan con el mismo pipeline de exportación (sin grafo TF a
    mano). Fixtures en `ml/fixtures/dummy/`.
- [x] **T-B003-10 · Exportación al contrato** (`OutputContractWrapper`, float16 e int8)
  - Req.: REQ-B003-18.
  - Nota de implementación (ADR-B003-07): `torch.onnx.export` (opset 17, exportador clásico) → `onnx2tf` →
    `.tflite` float16/float32; int8 con el conversor de TF y un dataset representativo. La salida se renombra a
    `input`/`output` y se comprueba que solo hay *builtins*. `onnx2tf` intenta descargar un fichero de muestra de
    GitHub: el pipeline lo crea en el directorio de trabajo para que funcione sin red.
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
