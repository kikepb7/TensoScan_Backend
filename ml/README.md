# tensoscan-ml

Herramientas de ML de TensoScan (spec B-003): datos sintéticos, ingesta, entrenamiento, evaluación y exportación a LiteRT
del detector de glifos de 7 segmentos. Paquete independiente: el servicio web no lo importa (C-ML-01).

```bash
cd ml
uv sync --group dev                                   # núcleo ligero
uv sync --group dev --extra train --extra export      # + PyTorch, ONNX, onnx2tf, TensorFlow, LiteRT
uv run tensoscan-ml --help
```

(La documentación completa se amplía en las tareas siguientes de B-003.)
