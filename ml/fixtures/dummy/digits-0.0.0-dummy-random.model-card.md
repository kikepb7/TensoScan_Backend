# Model card: digit-detector 0.0.0-dummy-random

> Plantilla generada por `tensoscan-ml export`. Los campos marcados con **TODO** los completa quien publica el modelo;
> `tensoscan-ml release` no permite publicar mientras queden **TODO** sin resolver.

## Resumen

| Campo | Valor |
|-------|-------|
| Nombre | `digit-detector` |
| Versión | `0.0.0-dummy-random` |
| Versión del contrato | `1.0.0` |
| Fichero | `digits-0.0.0-dummy-random.tflite` |
| sha256 | `cc209996db510bf2bd915d210673e3bf0de2ff1772cc51a74ee06f7a94b5fd98` |
| Entrada | `[1, 320, 320, 3]` float32, RGB, letterbox con relleno gris 0.447 |
| Salida | `[1, 2100, 18]`: `cx, cy, w, h` normalizados + 14 puntuaciones sigmoide, sin NMS |
| Tamaño | 1435 KiB |
| Operaciones | ADD, CONCATENATION, CONV_2D, DEPTHWISE_CONV_2D, DEQUANTIZE, EXP, LOGISTIC, MAX_POOL_2D, MINIMUM, MUL, PAD, RELU, RESHAPE, RESIZE_NEAREST_NEIGHBOR, STRIDED_SLICE, TRANSPOSE |
| Commit | `dummy` |

## Uso previsto

Detectar los glifos de 7 segmentos del display de un tensiómetro dentro del recorte guía de la app. La asignación a
SYS/DIA/PUL la hace el parser determinista de `reading-rules.md`. **TODO**: confirmar el uso previsto.

## Datos de entrenamiento

- Versión del dataset: `dummy-random` (hash en `run.json` del entrenamiento).
- Sintéticos: generador `tensoscan-ml synth` (semilla y configuración en el dataset).
- Reales: **TODO** (dispositivos, número de lecturas y de fotogramas, condiciones de luz, consentimiento).

## Métricas

| Métrica | Valor |
|---------|-------|
| Acierto exacto de la lectura (test real) | **TODO** |
| mAP@0.5 sintético (validación) | **TODO** |
| mAP@0.5 real (test) | **TODO** |

**TODO**: indicar si la evaluación es `beta` (dispositivos vistos) o `production` (dispositivos no vistos).

## Límites conocidos

- El parser compartido no une un «1» a su vecino izquierdo con fuentes anchas (hallazgo F1 del README de `ml/`).
- Quedan fuera de alcance los displays de matriz de puntos o no de 7 segmentos y los de muñeca con el display rotado.
- **TODO**: completar con los fallos observados en la evaluación.

## Sesgos de dispositivo

**TODO**: marcas y tipos de LCD vistos y no vistos.

## Licencias

El código de entrenamiento y la arquitectura son originales del proyecto; no se usa código ni pesos AGPL/GPL.
