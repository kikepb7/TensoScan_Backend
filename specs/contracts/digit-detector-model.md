# Contrato del modelo `digit-detector` (LiteRT)

> **Fuente canónica**: `TensoScan_Backend/specs/contracts/digit-detector-model.md`.
> Copia sincronizada en `TensoScan_App/specs/contracts/digit-detector-model.md`.
> Productor: `TensoScan_Backend/ml` (spec B-003). Consumidor: `LiteRtDigitDetector` de la app
> (spec A-002).
>
> Versión del contrato: **1.0.0** · Estado: 📝 Pendiente de aprobación

## 1. Propósito

Detectar los **glifos de 7 segmentos** encendidos dentro del recorte del recuadro guía (ROI).
El modelo no sabe qué número es SYS, DIA o PUL: esa asignación la hace el parser determinista
de `reading-rules.md`. Así el modelo **no depende del modelo de tensiómetro** y el layout se
resuelve con reglas que se pueden testear.

## 2. Artefactos de cada versión

```
digits-<semver>.tflite        # modelo LiteRT (float16 por defecto; variante -int8 opcional)
digits-<semver>.json          # metadatos sidecar (ver §5)
digits-<semver>.model-card.md # datos de entrenamiento, métricas, límites conocidos
```
Versionado semántico: **MAJOR** cambia la E/S o las clases (incompatible con la app); **MINOR**
reentrena con la misma E/S; **PATCH** corrige metadatos.

## 3. Entrada

| Propiedad | Valor |
|-----------|-------|
| Tensor | `input`, shape `[1, 320, 320, 3]`, layout NHWC, orden de canales **RGB** |
| Tipo | `float32` en `[0, 1]` (modelo float16) · `uint8` en `[0, 255]` (variante int8, con los parámetros de cuantización del tensor) |
| Preprocesado (lo hace la app) | 1) Recortar la ROI del frame ya rotado a la orientación natural. 2) **Letterbox**: escalar manteniendo la proporción hasta que el lado mayor mida 320 y centrar con relleno gris `114/255`. 3) Normalizar dividiendo entre 255 |

## 4. Salida

| Propiedad | Valor |
|-----------|-------|
| Tensor | `output`, shape `[1, N, 4 + C]`, `float32` |
| `N` | Número fijo de anclas o propuestas (lo declara el sidecar; por ejemplo 2100 a 320 px) |
| Columnas 0–3 | Caja `cx, cy, w, h` **normalizada a la entrada de 320×320** (rango `[0, 1]`, en espacio letterbox) |
| Columnas 4… | `C` puntuaciones por clase, **ya pasadas por sigmoide**, en `[0, 1]` |
| Sin NMS | La app aplica umbral y NMS (ver §6). Así se evitan operaciones no soportadas por LiteRT o por el GPU delegate |

### 4.1 Clases (orden fijo, `C = 14`)

| Índice | Etiqueta | `DisplaySymbol` (app) | Notas |
|--------|----------|-----------------------|-------|
| 0–9 | `0`…`9` | `DIGIT_0`…`DIGIT_9` | Dígitos de 7 segmentos |
| 10 | `colon` | `COLON` | Separador de la hora (permite descartar el reloj) |
| 11 | `dash` | `DASH` | `---` de medición en curso o en error |
| 12 | `E` | `LETTER_E` | Códigos de error (`E1`, `Err`) |
| 13 | `r` | `LETTER_R` | Parte de `Err` |

## 5. Sidecar `digits-<semver>.json`

```json
{
  "name": "digit-detector",
  "version": "1.0.0",
  "contract_version": "1.0.0",
  "input": { "width": 320, "height": 320, "channels": 3, "dtype": "float32", "range": [0, 1], "letterbox_pad": 0.447 },
  "output": { "num_candidates": 2100, "box_format": "cxcywh_normalized", "scores": "sigmoid" },
  "classes": ["0","1","2","3","4","5","6","7","8","9","colon","dash","E","r"],
  "recommended": { "score_threshold": 0.40, "nms_iou_threshold": 0.45, "max_detections": 32 },
  "sha256": "<hash del .tflite>",
  "training": { "dataset_version": "ds-2026.11.0", "git_commit": "<sha>" },
  "metrics": { "reading_exact_match_real_test": 0.0, "map50_synthetic_val": 0.0, "map50_real_test": 0.0 }
}
```
La app **verifica** `contract_version` (major compatible) y `sha256` al cargar el modelo. Si no
coinciden, el detector se considera no disponible y se usa el *fallback* (ver A-002
ADR-A002-03).

## 6. Postprocesado (lo hace la app)

1. Para cada candidato: `class = argmax(scores)` y `score = max(scores)`. Se descarta si
   `score < score_threshold`.
2. NMS **agnóstico de clase** con IoU `nms_iou_threshold`: los glifos de un display no se
   solapan, así que dos cajas solapadas son la misma detección.
3. Se conservan como máximo `max_detections`, ordenadas por score.
4. Se deshace el letterbox → coordenadas normalizadas **a la ROI** (`[0, 1]`). Esas son las
   coordenadas que consume el parser.

## 7. Presupuesto no funcional

| Métrica | Objetivo |
|---------|----------|
| Tamaño del `.tflite` | ≤ 6 MB (float16) |
| Latencia de inferencia p50 (dispositivo de referencia, CPU XNNPACK 4 hilos) | ≤ 25 ms |
| Latencia p95 | ≤ 45 ms |
| Operaciones | Solo *builtins* de LiteRT (sin Flex ni custom ops) |

Dispositivo de referencia: gama media de 2023 o posterior (por ejemplo Pixel 6a o Galaxy A54).
Se mide además en un dispositivo de gama baja con Android 9 (minSdk 28).
