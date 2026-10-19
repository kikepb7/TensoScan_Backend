# Model card: digit-detector 0.0.0-dummy-fixed

> Plantilla generada por `tensoscan-ml export`. Los campos marcados con **TODO** los completa quien publica el modelo;
> `tensoscan-ml release` no permite publicar mientras queden **TODO** sin resolver.

## Resumen

| Campo | Valor |
|-------|-------|
| Nombre | `digit-detector` |
| Versión | `0.0.0-dummy-fixed` |
| Versión del contrato | `1.0.0` |
| Fichero | `digits-0.0.0-dummy-fixed.tflite` |
| sha256 | `13b600151bef0fadd4a5379250650fffc020f27d623d16e5597ca834be86e319` |
| Entrada | `[1, 320, 320, 3]` float32, RGB, letterbox con relleno gris 0.447 |
| Salida | `[1, 2100, 18]`: `cx, cy, w, h` normalizados + 14 puntuaciones sigmoide, sin NMS |
| Tamaño | 75 KiB |
| Operaciones | ADD, DEQUANTIZE, MEAN, MUL |
| Commit | `dummy` |

## Uso previsto

Detectar los glifos de 7 segmentos del display de un tensiómetro dentro del recorte guía de la app. La asignación a
SYS/DIA/PUL la hace el parser determinista de `reading-rules.md`. **TODO**: confirmar el uso previsto.

## Datos de entrenamiento

- Versión del dataset: `dummy-fixed` (hash en `run.json` del entrenamiento).
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
