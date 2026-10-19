# tensoscan-ml

Herramientas de ML de TensoScan (spec B-003): datos sintéticos, ingesta de capturas reales, entrenamiento, evaluación y
exportación a LiteRT del **detector de glifos de 7 segmentos** que usa la app. Es un paquete independiente: el servicio web
no lo importa (C-ML-01) y los datos reales no se versionan nunca (solo los manifiestos, C-ML-03).

- Contrato del modelo: [`specs/contracts/digit-detector-model.md`](../specs/contracts/digit-detector-model.md)
- Reglas de lectura (parser y validador): [`specs/contracts/reading-rules.md`](../specs/contracts/reading-rules.md)
- Spec: [`specs/003-digit-recognition-model/`](../specs/003-digit-recognition-model/)

## 1. Instalación

Requiere Python ≥ 3.12 y [`uv`](https://docs.astral.sh/uv/).

```bash
cd ml
uv sync --group dev                                      # núcleo ligero: parser, sintético, ingesta, eval, tests rápidos
uv sync --group dev --extra train                        # + PyTorch (entrenar, evaluar un checkpoint)
uv sync --group dev --extra train --extra export         # + ONNX, onnx2tf, TensorFlow, LiteRT (exportar)
uv sync --group dev --extra runtime                      # solo el intérprete LiteRT (verify, eval de un .tflite)
uv run tensoscan-ml --help
```

Notas:

- En Linux, `torch` de PyPI trae ruedas con CUDA (~3 GB). Para solo CPU, tras `uv sync`:
  `uv pip install --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cpu`.
  En Colab/Kaggle se deja la rueda por defecto (usa la GPU).
- Los datos viven **fuera de git**: exporta `TENSOSCAN_ML_DATA_DIR=/ruta/a/los/datos` (por defecto `ml/data/`, ignorado).
- Puerta de calidad local: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest`.
  Los tests que necesitan PyTorch o el stack de exportación llevan las marcas `torch`, `export` y `runtime` y se omiten si
  el extra no está instalado; el CI rápido (`ml-tests`) ejecuta el resto, y `ml-tests-heavy` ejecuta todo.

## 2. Comandos

Todos los comandos escriben un `run.json` con la semilla, el commit, la versión y el hash del dataset y la configuración
(REQ-B003-02). Las configuraciones están en `configs/` y se pueden ajustar con `--config mi.yaml` y `--set a.b=valor`.

| Comando | Qué hace |
|---------|----------|
| `synth` | Genera imágenes sintéticas con etiquetas YOLO exactas (`--seed`, `--train/--val/--test`, `--workers`, `--gallery`) |
| `ingest` | Vídeos y fotos (nombre convenido o `labels.csv`) → fotogramas recortados, sin borrosos ni duplicados, + manifiesto |
| `split` | Reparte un manifiesto en train/val/test sin fugas (por dispositivo o por lectura) |
| `prelabel` | Pre-anota ~300 fotogramas con un modelo (YOLO + zip para CVAT) o `--check` de las anotaciones corregidas |
| `train` | Entrenamiento en dos etapas (sintético, luego fine-tuning con real); checkpoints, `metrics.csv`, `curves.png` |
| `eval` | Evalúa un checkpoint, un `.tflite` o un `detections.jsonl` con el parser oficial; `report.md`/`report.json` |
| `export` | Checkpoint → ONNX → `.tflite` (float16, int8 opcional) + sidecar JSON + plantilla de model card |
| `verify` | Comprueba un `.tflite` contra el contrato (E/S, builtins, clases, tamaño, latencia, set dorado) |
| `dummy-model` | Genera modelos de pruebas que respetan el contrato (pesos aleatorios, o lectura fija) |
| `release` | Esqueleto: lo implementa T-B003-13 (necesita datos reales y decisiones del usuario) |
| `audit-licenses` | Falla si una dependencia instalada es GPL/AGPL |

### Datos sintéticos

```bash
uv run tensoscan-ml synth --out $TENSOSCAN_ML_DATA_DIR/synth/ds-synth-0.1.0 --seed 42 --workers 4   # 30.000 + 3.000
uv run tensoscan-ml synth --gallery docs/synth-samples.png                                            # 50 muestras para revisar
```

La salida es una función pura de `(config, semilla)`: dos ejecuciones dan ficheros idénticos byte a byte, con cualquier
número de procesos. `meta.jsonl` guarda por imagen la plantilla, los valores, los parámetros de render y si el parser
oficial reproduce la lectura a partir de las etiquetas exactas.

### Tus fotos con `labels.csv` (un tensiómetro, varias lecturas)

1. Pon las fotos y un `labels.csv` en una carpeta (formato y reglas en [`docs/capture-protocol.md`](docs/capture-protocol.md)):

   ```csv
   file,device,sys,dia,pul,condition
   IMG_0001.jpg,omron-m3,128,82,67,good
   IMG_0002.jpg,omron-m3,141,94,78,glare
   ```

2. Comprueba el encuadre y procesa:

   ```bash
   export TENSOSCAN_ML_DATA_DIR=/ruta/a/los/datos
   uv run tensoscan-ml ingest --src /ruta/a/las/fotos --dataset-version ds-real-0.1.0 --roi-mode center --preview
   #   abre $TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0/preview.jpg; si no encuadra el display usa --roi-mode none|manual
   uv run tensoscan-ml ingest --src /ruta/a/las/fotos --dataset-version ds-real-0.1.0 --roi-mode center --commit-manifest
   uv run tensoscan-ml split --manifest $TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0/manifest.jsonl
   ```

3. Evalúa el modelo en el test real y, si quieres cajas reales, pre-anota y corrige en CVAT
   ([`docs/cvat-annotation-guide.md`](docs/cvat-annotation-guide.md)):

   ```bash
   D=$TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0
   uv run tensoscan-ml eval --detector tflite --model dist/digits-0.1.0.tflite --manifest $D/test.jsonl
   uv run tensoscan-ml prelabel --manifest $D/manifest.jsonl --detector tflite --model dist/digits-0.1.0.tflite --out $D/annotate
   ```

Con un solo dispositivo el informe marca la evaluación como `beta` (cifras optimistas); `production` exige ≥ 3
dispositivos con uno excluido del entrenamiento.

### Entrenar, exportar y verificar

```bash
uv run tensoscan-ml train --config configs/train/yolox-nano-v1.yaml     # etapa 2 se omite (con aviso) sin datos reales con cajas
uv run tensoscan-ml export --checkpoint $TENSOSCAN_ML_DATA_DIR/runs/yolox-nano-v1/pretrain-synthetic/best.pt --out dist --version 0.1.0
uv run tensoscan-ml verify --model dist/digits-0.1.0.tflite --golden synthetic
uv run tensoscan-ml eval --detector tflite --model dist/digits-0.1.0.tflite --manifest $TENSOSCAN_ML_DATA_DIR/synth/ds-synth-0.1.0/val/manifest.jsonl
```

Prueba de humo en CPU (segundos): `uv run tensoscan-ml train --config configs/train/smoke.yaml`.

### Dónde encaja Colab / Kaggle

El entrenamiento completo (1–3 h por entrenamiento con 30.000 imágenes) se hace en una GPU gratuita. El cuaderno
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) clona el repositorio, instala los extras, monta Drive para los
datos y ejecuta `synth → train → export → verify`. En Kaggle es igual: activa la GPU, instala con `uv sync` y apunta
`TENSOSCAN_ML_DATA_DIR` a `/kaggle/working`. Los pesos y artefactos no van a git: se publican como *assets* de una GitHub
Release.

## 3. Cambiar de arquitectura (ADR-B003-02)

El detector por defecto es una red *anchor-free* propia al estilo YOLOX-Nano (0,69 M de parámetros, ~1,4 MB en float16),
escrita desde cero con licencia del proyecto: **no hay código ni pesos de Ultralytics/AGPL**. Para probar otra candidata
(YOLOX-Nano oficial, NanoDet-Plus) se registra un constructor en `tensoscan_ml/model/yolox.py::ARCHITECTURES` que devuelva un
módulo con la misma interfaz (una lista de mapas `[B, 4 + 1 + C, H, W]` por nivel con strides 8/16/32). El resto del pipeline
(pérdida, wrapper de salida, exportación, verify, eval) no cambia. La elección final es el *spike* T-B003-08.

## 4. Modelos «dummy» para la app Android

Para desarrollar `LiteRtDigitDetector` antes de entrenar, `ml/fixtures/dummy/` contiene dos modelos **sin valor predictivo**
con la E/S exacta del contrato (`input [1,320,320,3] float32`, `output [1,2100,18] float32`, solo *builtins*):

| Fichero | Contenido | Tamaño |
|---------|-----------|--------|
| `digits-0.0.0-dummy-fixed.tflite` (+ `.json`) | Ignora la imagen y devuelve siempre la lectura **128/82/67** (ROI 4:5, apilado), más un duplicado (NMS) y una caja de score 0,30 (umbral) | 75 KB |
| `digits-0.0.0-dummy-random.tflite` (+ `.json`) | La arquitectura real con pesos aleatorios; el sesgo de las cabezas hace que todas las anclas pasen el umbral: obliga al NMS y al límite de 32 | 1,4 MB |

La app copia el `.tflite` y su `.json` a `data/src/main/assets/models/`, verifica `contract_version` (mayor 1) y el `sha256`
del sidecar (que es el del `.tflite`), y aplica el preprocesado (letterbox gris 114/255, RGB, `[0,1]`) y el postprocesado del
contrato. Con el modelo `fixed` el resultado esperado tras decodificar y pasar por el parser es `COMPLETE 128/82/67`.
Regenerar: `uv run tensoscan-ml dummy-model` (determinista: el mismo sha256).

## 5. Hallazgos para el contrato (no se ha modificado ningún contrato)

**F1 · El parser compartido no une un «1» a su vecino izquierdo con fuentes normales.** El hueco entre un dígito y un `1`
que le sigue es `ancho_celda + separación − grosor_del_trazo` (el `1` solo ocupa los segmentos de la derecha de su celda).
Con una fuente ordinaria (ancho 0,50·H, trazo 0,15·H, separación 0,12·H) eso es ≈ 0,47·H, y `GROUP_MAX_GAP_FACTOR` permite
0,35 de la altura **en coordenadas normalizadas a la ROI**, es decir 0,35 × (ancho/alto) = 0,28·H en píxeles para una ROI
4:5. Resultado medido con las cajas exactas, sin ruido: **solo el 54 % de las lecturas normales** (SYS 100–139, DIA 60–89,
PUL 55–95) se reconstruyen, y el 46 % restante son exactamente las que llevan un `1` después del primer dígito de un número
(`121`, `81`, `71`...). Ningún caso dorado lo cubre (todos sus `1` van primero). Los glifos `3` y `7` tras otro dígito tienen
el mismo problema en menor grado. **Propuesta probada**: para un `1`, relajar solo la cota superior del hueco con el ancho
de su celda virtual (`allowance = max(0, 0,55·h/aspecto − ancho)`); pasa los 14 casos dorados y reconstruye el 100 % de las
lecturas con las tres fuentes probadas. Requiere subir la versión del contrato y regenerar los casos. El generador marca
cada fuente como compatible o no (`parser_compatible` en `meta.jsonl`) y los tests documentan el comportamiento actual.

**F2 · Rotación y perspectiva.** Con rotaciones de ±8° las cajas alineadas a ejes de un `1` (alto y estrecho) se solapan con
las del dígito vecino más de lo que tolera `GROUP_MAX_OVERLAP_FACTOR` (0,30 × el ancho menor). Con fuentes compatibles y
perspectiva/rotación completas el parser reproduce ~86–98 % de las lecturas legibles, frente al 100 % sin ellas.

**F3 · `:` y `-` en los casos dorados tienen cajas de la altura de un dígito.** Una caja ajustada al guion mide ~0,15 de la
altura del dígito, por debajo de `MIN_SYMBOL_HEIGHT` (0,04 de la ROI) salvo displays muy grandes, así que `---` se
reconocería como `NO_READING` en lugar de `MEASURING`. La guía de CVAT pide cajas ajustadas; si se quiere mantener el
convenio de los casos dorados hay que decidirlo en el contrato.

**F4 · Operativo.** `onnx2tf` intenta descargar de GitHub un fichero de muestra de calibración al convertir; el pipeline lo
crea localmente para funcionar sin red. El escritor TFLite de `torch.onnx.export` clásico está deprecado en PyTorch ≥ 2.9
(`dynamo=False`); la versión queda fijada por `uv.lock`.

## 6. Estructura

```
configs/            synth, ingest (+split), train, export
docs/               protocolo de captura, guía de CVAT, galería sintética
fixtures/dummy/     modelos de pruebas para la app
notebooks/          train_colab.ipynb
datasets/manifests/ manifiestos versionados (sin imágenes)
tensoscan_ml/       reading, synth, data, detector, eval, model, export, verify, audit, commands
tests/
```
