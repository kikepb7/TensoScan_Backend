"""Sidecar JSON of a model (contract section 5) and the model-card template (REQ-B003-20)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensoscan_ml.contract import (
    CLASSES,
    CONTRACT_VERSION,
    INPUT_CHANNELS,
    INPUT_SIZE,
    MAX_DETECTIONS,
    MODEL_NAME,
    NMS_IOU_THRESHOLD,
    PAD_VALUE,
    SCORE_THRESHOLD,
)
from tensoscan_ml.provenance import file_sha256, git_commit


def build_sidecar(
    tflite_path: str | Path,
    *,
    version: str,
    num_candidates: int,
    int8: bool = False,
    dataset_version: str = "unknown",
    metrics: dict[str, float] | None = None,
    input_size: int = INPUT_SIZE,
) -> dict[str, Any]:
    """The `digits-<version>.json` content for the given `.tflite` (with its sha256)."""
    return {
        "name": MODEL_NAME,
        "version": version,
        "contract_version": CONTRACT_VERSION,
        "input": {
            "width": input_size,
            "height": input_size,
            "channels": INPUT_CHANNELS,
            "dtype": "uint8" if int8 else "float32",
            "range": [0, 255] if int8 else [0, 1],
            "letterbox_pad": round(PAD_VALUE, 3),
        },
        "output": {"num_candidates": num_candidates, "box_format": "cxcywh_normalized", "scores": "sigmoid"},
        "classes": list(CLASSES),
        "recommended": {
            "score_threshold": SCORE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS,
        },
        "sha256": file_sha256(tflite_path),
        "training": {"dataset_version": dataset_version, "git_commit": git_commit()},
        "metrics": {
            "reading_exact_match_real_test": 0.0,
            "map50_synthetic_val": 0.0,
            "map50_real_test": 0.0,
            **(metrics or {}),
        },
    }


def write_sidecar(path: str | Path, sidecar: dict[str, Any]) -> Path:
    target = Path(path)
    target.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


MODEL_CARD_TEMPLATE = """# Model card: digit-detector {version}

> Plantilla generada por `tensoscan-ml export`. Los campos marcados con **TODO** los completa quien publica el modelo;
> `tensoscan-ml release` no permite publicar mientras queden **TODO** sin resolver.

## Resumen

| Campo | Valor |
|-------|-------|
| Nombre | `{name}` |
| Versión | `{version}` |
| Versión del contrato | `{contract_version}` |
| Fichero | `{filename}` |
| sha256 | `{sha256}` |
| Entrada | `[1, {size}, {size}, 3]` {dtype}, RGB, letterbox con relleno gris {pad} |
| Salida | `[1, {candidates}, {values}]`: `cx, cy, w, h` normalizados + {classes_count} puntuaciones sigmoide, sin NMS |
| Tamaño | {size_kib:.0f} KiB |
| Operaciones | {operators} |
| Commit | `{commit}` |

## Uso previsto

Detectar los glifos de 7 segmentos del display de un tensiómetro dentro del recorte guía de la app. La asignación a
SYS/DIA/PUL la hace el parser determinista de `reading-rules.md`. **TODO**: confirmar el uso previsto.

## Datos de entrenamiento

- Versión del dataset: `{dataset_version}` (hash en `run.json` del entrenamiento).
- Sintéticos: generador `tensoscan-ml synth` (semilla y configuración en el dataset).
- Reales: **TODO** (dispositivos, número de lecturas y de fotogramas, condiciones de luz, consentimiento).

## Métricas

| Métrica | Valor |
|---------|-------|
| Acierto exacto de la lectura (test real) | {reading} |
| mAP@0.5 sintético (validación) | {map_synthetic} |
| mAP@0.5 real (test) | {map_real} |

**TODO**: indicar si la evaluación es `beta` (dispositivos vistos) o `production` (dispositivos no vistos).

## Límites conocidos

- El parser compartido no une un «1» a su vecino izquierdo con fuentes anchas (hallazgo F1 del README de `ml/`).
- Quedan fuera de alcance los displays de matriz de puntos o no de 7 segmentos y los de muñeca con el display rotado.
- **TODO**: completar con los fallos observados en la evaluación.

## Sesgos de dispositivo

**TODO**: marcas y tipos de LCD vistos y no vistos.

## Licencias

El código de entrenamiento y la arquitectura son originales del proyecto; no se usa código ni pesos AGPL/GPL.
"""


def render_model_card(sidecar: dict[str, Any], *, filename: str, size_bytes: int, operators: list[str]) -> str:
    metrics = sidecar["metrics"]

    def pct(value: float) -> str:
        return "**TODO**" if not value else f"{100 * value:.1f} %"

    return MODEL_CARD_TEMPLATE.format(
        version=sidecar["version"],
        name=sidecar["name"],
        contract_version=sidecar["contract_version"],
        filename=filename,
        sha256=sidecar["sha256"],
        size=sidecar["input"]["width"],
        dtype=sidecar["input"]["dtype"],
        pad=sidecar["input"]["letterbox_pad"],
        candidates=sidecar["output"]["num_candidates"],
        values=4 + len(sidecar["classes"]),
        classes_count=len(sidecar["classes"]),
        size_kib=size_bytes / 1024,
        operators=", ".join(sorted(set(operators))),
        commit=sidecar["training"]["git_commit"],
        dataset_version=sidecar["training"]["dataset_version"],
        reading=pct(metrics["reading_exact_match_real_test"]),
        map_synthetic=pct(metrics["map50_synthetic_val"]),
        map_real=pct(metrics["map50_real_test"]),
    )
