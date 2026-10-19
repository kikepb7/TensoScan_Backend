# B-003 — Modelo de detección de dígitos · Diseño

> Estado: 📝 Pendiente de aprobación · Requisitos: [requirements.md](requirements.md)

## 1. Estructura

```
ml/
├── pyproject.toml                 # tensoscan-ml; deps: numpy, opencv-python-headless, pillow, pyyaml,
│                                  # imagehash, torch (+ arquitectura elegida), onnx, onnx2tf, ai-edge-litert
├── configs/
│   ├── synth/default.yaml         # parámetros del generador
│   ├── train/<arch>-v1.yaml       # hiperparámetros, etapas y rutas de datos
│   └── export/v1.yaml
├── tensoscan_ml/
│   ├── cli.py
│   ├── reading/                   # parser.py, validator.py (port oficial de reading-rules)
│   ├── synth/
│   │   ├── glyph.py               # geometría de 7 segmentos → polígonos por segmento
│   │   ├── fonts.py               # estilos (grosor, inclinación, biselado, fantasma)
│   │   ├── layouts.py             # plantillas de pantalla + elementos auxiliares
│   │   ├── scene.py               # display + carcasa + fondo + ROI
│   │   ├── augment.py             # perspectiva, blur, ruido, reflejos, sombras, JPEG...
│   │   └── writer.py              # imágenes + etiquetas YOLO + metadatos JSONL
│   ├── data/
│   │   ├── naming.py              # parseo y validación del nombre de fichero
│   │   ├── ingest.py              # vídeo → frames → calidad → dedupe → ROI → manifiesto
│   │   ├── split.py               # división agrupada por dispositivo
│   │   └── prelabel.py            # export CVAT/YOLO
│   ├── model/                     # adaptador de la arquitectura elegida (train/predict)
│   ├── export/                    # wrapper de salida → ONNX → onnx2tf → .tflite, int8
│   ├── eval/                      # detecciones → parser → métricas → informes
│   └── release/                   # sidecar, model card, verificación de umbrales
├── datasets/manifests/            # *.jsonl versionados (sin imágenes)
└── tests/
```

## 2. Decisiones (ADR)

### ADR-B003-01 — Detector de glifos (y no clasificador ni OCR de secuencia)
Es coherente con App ADR-A002-02: el layout lo resuelve el parser compartido. Un detector de
glifos:
- no depende del número de dígitos ni de su posición;
- detecta también `:`, `-`, `E` y `r`, que son necesarios para filtrar la hora y los errores;
- tiene métricas interpretables por clase.

Un reconocedor de secuencia (CRNN + CTC) necesitaría separar antes las líneas, lo que equivale a
un segundo detector.

### ADR-B003-02 — Arquitectura: decisión por *spike* con criterios explícitos

| Candidata | Licencia | Pros | Contras |
|-----------|----------|------|---------|
| **YOLOX-Nano** | Apache-2.0 | Muy pequeña (~0,9 M parámetros), *anchor-free*, buen rendimiento con objetos pequeños | Exportación vía ONNX → onnx2tf (hay que validar las operaciones) |
| **NanoDet-Plus-m** | Apache-2.0 | Diseñada para móvil, muy rápida | Mismo camino de exportación; comunidad menor |
| Ultralytics YOLO (v8n/11n) | **AGPL-3.0** | Mejor herramienta y exportación a TFLite integrada | Obliga a licenciar la app bajo AGPL o comprar una licencia Enterprise (Q-B003-01) |
| EfficientDet-Lite0 (MediaPipe Model Maker) | Apache-2.0 | Exportación TFLite nativa | Menos precisa con objetos pequeños, su postprocesado usa una operación propia y el mantenimiento de Model Maker es incierto |

**Criterios del *spike* (T-B003-08)**, por orden:
1. licencia compatible;
2. el `.tflite` solo usa *builtins* y pasa `verify`;
3. mAP@0.5 en la validación sintética tras el mismo presupuesto de entrenamiento;
4. latencia en CPU;
5. sencillez del pipeline.

**Opción por defecto** si el *spike* no es concluyente: **YOLOX-Nano**.

### ADR-B003-03 — Contrato de salida dentro del grafo
Se añade un *wrapper* (módulo PyTorch) tras la cabeza del detector que:
1. decodifica la rejilla y los *strides* a `cx, cy, w, h` normalizados a 320;
2. aplica sigmoide a las puntuaciones de clase (multiplicadas por el *objectness* si la
   arquitectura lo tiene);
3. concatena todo en `[1, N, 4 + 14]`.

La NMS queda fuera (contrato §4). Así la app no conoce la arquitectura, y cambiar de modelo no
cambia el código Kotlin mientras se respete el contrato.

### ADR-B003-04 — Primero datos sintéticos y después datos reales, con la evaluación siempre en real
- El sintético da volumen y etiquetas perfectas. El real cierra la brecha de dominio.
- La métrica de aceptación **solo** se mide en real, con dispositivos no vistos, para no
  engañarse con el sintético.

### ADR-B003-05 — Entrada de 320×320
Con la ROI de la app (unos 576×720 px antes de reescalar), los dígitos principales de un display
encuadrado ocupan entre el 15 y el 30 % de la altura de la ROI, es decir, entre 48 y 96 px a 320.
Es suficiente para un detector *nano*. 416 se evaluaría solo si fallan los glifos pequeños del
pulso (≥ 8 % de la altura, unos 25 px).

### ADR-B003-06 — Etiquetado sin cajas para la evaluación
La lectura correcta va **en el nombre del fichero**, así que el acierto por lectura (la métrica
principal) se mide sin dibujar cajas. Las cajas solo se dibujan (con pre-etiquetado) en un
subconjunto de ~300 frames reales, para el *fine-tuning* y el mAP real.

## 3. Generador sintético

### 3.1 Glifo de 7 segmentos (`glyph.py`)

```
 ─a─
|f  |b      Segmentos a–g como polígonos hexagonales (extremos biselados) o rectángulos con esquinas
 ─g─        redondeadas. Parámetros: ancho W, alto H, grosor t (0,10–0,22·H), hueco entre
|e  |c      segmentos (0,01–0,04·H), inclinación (0–12°).
 ─d─        dígito → segmentos encendidos (tabla estándar); colon = 2 puntos; dash = g; E = a,d,e,f,g; r = e,g
```
- Segmentos apagados (fantasma): se pintan con una opacidad α ∈ [0, 0,3]. Es lo que confunde al
  OCR genérico y aquí se entrena explícitamente.
- La caja de la etiqueta es la envolvente de los segmentos **encendidos** con el mismo margen que
  en real (+4 % de H). El dígito `1` es estrecho y su caja también.

### 3.2 Plantillas de layout (`layouts.py`)

| Plantilla | Proporción en el set |
|-----------|----------------------|
| SYS, DIA y PUL apilados con PUL pequeño a la derecha | 35 % |
| Tres filas de igual altura | 15 % |
| SYS \| DIA en la misma fila y PUL debajo | 15 % |
| Pantalla con hora o fecha visibles | 10 % (combinable) |
| `E1`, `Err` y `---` | 10 % |
| Negativos: pantalla apagada o carcasa sin display | 5 % |
| Fuera de cobertura (para robustez de glifos): números sueltos o 4 dígitos | 10 % |

- Los valores se muestrean en 50 % rangos normales, 30 % extremos válidos y 20 % aleatorios
  (00–299). El modelo aprende glifos, no semántica.
- Elementos auxiliares (texto «SYS», «mmHg», iconos) con fuentes sans **no** de 7 segmentos:
  son **distractores sin etiqueta**.

### 3.3 Escena y aumentos (`scene.py`, `augment.py`)
- Carcasa (color plano o textura plástica) con el display incrustado y biselado. El display ocupa
  entre el 40 y el 95 % de la ROI.
- Fondos: texturas (mesa, tela y madera) procedurales o de un banco libre de derechos (CC0)
  descargado por un script con checksums.
- Pipeline (probabilidades en `configs/synth/default.yaml`):
  1. perspectiva de ±25° y rotación de ±8°;
  2. reflejo especular (elipse con gradiente, α de 0,2–0,9);
  3. sombra parcial;
  4. gamma de 0,5–1,8;
  5. balance de blancos;
  6. desenfoque de movimiento (k de 3–9) o gaussiano;
  7. ruido de Poisson + gaussiano;
  8. JPEG con calidad de 40–95;
  9. reescalado a la resolución de la ROI de la app (≤ 640) y vuelta.
- Volumen inicial: 30.000 de entrenamiento y 3.000 de validación (configurable).

## 4. Protocolo de captura de datos reales (para el usuario)

**Material**: tensiómetros disponibles (Q-B003-02) y el móvil con la cámara normal.

**Por cada tensiómetro y cada medición real que se haga**:
1. Hacer la medición normal en el aparato. **Anotar los tres valores** que muestra.
2. Grabar un **vídeo de 15–30 s** apuntando al display, moviendo el móvil despacio (acercar y
   alejar, inclinar un poco a cada lado). No hace falta que salga perfecto: los defectos son
   útiles.
3. Repetir en 2–3 condiciones de luz cuando sea posible: `good` (luz natural o lámpara), `dim`
   (poca luz) y `glare` (con un reflejo de lámpara o ventana sobre la pantalla).
4. Nombrar el fichero así:
   ```
   <marca>-<modelo>__<sys>-<dia>-<pul>__<condición>__<nn>.mp4
   ejemplo: omron-m3__128-82-67__dim__01.mp4
   ```
   Si el display muestra un error, se escribe `error` en lugar de los valores
   (`omron-m3__error__good__01.mp4`).
5. Objetivo mínimo para v1: **≥ 3 modelos** de tensiómetro × **≥ 10 lecturas distintas** ×
   **2 condiciones** ≈ 60 vídeos (≈ 3.000 frames útiles tras filtrar).
6. Entrega: una carpeta compartida (Drive) o un disco. **No se suben al repositorio.**
7. Consentimiento: las personas que aparezcan o cuyos valores se registren deben consentirlo. Es
   preferible encuadrar **solo el display**, sin la persona.

## 5. Ingesta

```
ingest --src <dir> --out $TENSOSCAN_ML_DATA_DIR/real/<dataset_version>
  por fichero: naming.parse() → si es vídeo: frames a 3 fps (opencv)
    → calidad: varianza del laplaciano ≥ umbral (calibrable; por defecto 60)
    → dedupe: pHash, distancia de Hamming ≤ 4 dentro del mismo vídeo
    → ROI de la app: recorte centrado (ancho 0,80·W, alto min(1,25·ancho, 0,60·H), centro vertical 0,45·H)
       (misma geometría que App RoiGeometry §6.1) y reescalado a ≤ 640
    → manifest.jsonl: {path, sha256, device, sys, dia, pul|null, status, condition, video_id, frame_idx}
```

## 6. Evaluación

```
eval --detector {checkpoint|tflite|detections.jsonl} --manifest test.jsonl
  por frame: detecciones (contrato §6: umbral 0,40, NMS 0,45, máx. 32, coordenadas ROI)
           → tensoscan_ml.reading.parse() → comparación con la etiqueta
  métricas: acierto por frame (solo frames cuya etiqueta es lectura), acierto por vídeo
            (moda de los frames COMPLETE), desglose por dispositivo, condición y estado
            (DEVICE_ERROR acertado), mAP@0.5 y mAP@0.5:0.95 (si hay cajas) y matriz de
            confusión de glifos
  salida: report.md (tablas) + report.json (para release) + ejemplos de fallo (las 20 peores imágenes)
```
`--detector detections.jsonl` permite evaluar **cualquier** fuente, incluidas detecciones
volcadas por la app con ML Kit en modo debug, para comparar el *baseline* con la misma métrica.

## 7. Exportación y verificación

1. `export`: *checkpoint* → `OutputContractWrapper` (ADR-B003-03) → ONNX (opset fijo) → `onnx2tf`
   → `.tflite` float16 y, opcionalmente, int8 con 300 imágenes representativas (mezcla de real y
   sintético).
2. `verify`: `ai_edge_litert.Interpreter` → comprobaciones de REQ-B003-19.
   - La lista de operaciones se obtiene con el analizador de modelos de LiteRT/TF y **falla** si
     aparece `Flex` o `Custom`.
   - El set dorado de imágenes (20 frames reales y sintéticos con lectura conocida) vive en
     `ml/tests/golden_images/` **solo si** las imágenes son sintéticas o se tiene consentimiento
     explícito. Si no, se referencia por sha256 en el directorio de datos.
3. `release`: comprueba los umbrales de `report.json`, escribe el sidecar con las métricas reales y
   el sha256, genera `model-card.md` (datos, métricas, límites conocidos, sesgos de dispositivo
   y uso previsto) y empaqueta `dist/digits-<v>/`. La subida a GitHub Releases es manual o con
   `gh` en CI.

## 8. Riesgos

| Riesgo | Mitigación |
|--------|------------|
| Brecha entre sintético y real | Aumentos agresivos, *fine-tuning* real y evaluación solo en real con dispositivo excluido |
| Pocos dispositivos reales | Priorizar diversidad (marca y tipo de LCD) frente a cantidad; el sintético cubre la variedad de fuentes |
| Operaciones no soportadas al convertir | Criterio 2 del *spike*; arquitectura sin operaciones exóticas; `verify` en CI de la release |
| Fuga de datos entre train y test (frames del mismo vídeo) | División agrupada por dispositivo y vídeo |
| Licencias | ADR-B003-02 y Q-B003-01 |
