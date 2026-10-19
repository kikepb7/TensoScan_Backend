# B-003 — Modelo de detección de dígitos · Requisitos

> Estado: 📝 Pendiente de aprobación · Tipo: ML
> Produce: artefactos según [`digit-detector-model.md`](../contracts/digit-detector-model.md)
> Consume: [`reading-rules.md`](../contracts/reading-rules.md) (parser oficial en Python para evaluar)
> Habilita: App A-002 fase E · Resuelve: AUD-B-06

## 1. Contexto y objetivo

El modelo actual (clasificador 28×28 tras recortes fijos) no generaliza (AUD-B-01…06). La app
necesita un **detector de glifos de 7 segmentos** pequeño (≤ 6 MB), rápido en móvil (≤ 25 ms p50)
e independiente del modelo de tensiómetro, cuya salida convierte en SYS/DIA/PUL el parser
determinista compartido.

**Objetivo medible (v1.0.0)**: sobre el set de test **real** (dispositivos no vistos en el
entrenamiento):

| Métrica | Umbral de aceptación |
|---------|----------------------|
| Acierto exacto de la lectura **por frame** (SYS, DIA y PUL correctos tras el parser) | ≥ 90 % |
| Acierto exacto **por vídeo** (voto mayoritario de los frames `COMPLETE`) | ≥ 97 % |
| Acierto por dispositivo (peor dispositivo, por frame) | ≥ 85 % |
| mAP@0.5 de los glifos (subconjunto real con cajas) | ≥ 0,90 |
| mAP@0.5 de los glifos (validación sintética) | ≥ 0,95 |
| Tamaño, latencia y operaciones | Según el contrato §7 |

## 2. Historias

- **HU-B003-1**: Como ingeniero de ML, quiero generar miles de imágenes sintéticas etiquetadas
  automáticamente para entrenar sin etiquetar a mano.
- **HU-B003-2**: Como responsable del producto, quiero grabar vídeos de mis tensiómetros
  siguiendo un protocolo simple y que sirvan para evaluar sin dibujar cajas.
- **HU-B003-3**: Como desarrollador de la app, quiero un `.tflite` con E/S estable y verificada, y
  métricas reales antes de integrarlo.

## 3. Requisitos (EARS)

### Paquete y reproducibilidad
- **REQ-B003-01**: El código ML DEBERÁ vivir en `ml/` como paquete `tensoscan_ml`, con su
  `pyproject.toml` y su lock propios. El servicio web NO DEBERÁ importarlo.
- **REQ-B003-02**: Cada ejecución (generación, entrenamiento, evaluación o exportación) DEBERÁ
  configurarse con un YAML versionado y registrar la semilla, el commit, la versión del dataset y
  su hash.
- **REQ-B003-03**: DEBERÁ existir una CLI `tensoscan-ml` con los subcomandos `synth`, `ingest`,
  `split`, `prelabel`, `train`, `eval`, `export`, `verify` y `release`.

### Reglas compartidas
- **REQ-B003-04**: `tensoscan_ml.reading` DEBERÁ implementar el parser (§4) y el validador
  (§2–3) de `reading-rules.md`, y pasar **todos** los casos dorados.

### Datos sintéticos
- **REQ-B003-05**: El generador DEBERÁ renderizar glifos de 7 segmentos de forma paramétrica
  (grosor, inclinación de 0–12°, separación de segmentos, extremos biselados o redondeados,
  segmentos apagados «fantasma» con opacidad de 0–0,3) para todas las clases del contrato §4.1.
- **REQ-B003-06**: El generador DEBERÁ componer **pantallas completas** a partir de plantillas de
  layout (las de `reading-rules.md` §4.5 y alguna fuera de cobertura como negativo), con
  elementos auxiliares (etiquetas SYS/DIA/PUL, mmHg, icono de corazón, batería, «M», hora, fecha
  e índice de memoria) y estilos LCD (gris o verde con segmentos oscuros, retroiluminado) y LED.
- **REQ-B003-07**: El generador DEBERÁ aplicar aumentos que imiten la ROI de la app:
  - el display ocupa entre el 40 y el 95 % del recorte, con la carcasa visible alrededor;
  - perspectiva de ±25° y rotación de ±8°;
  - desenfoque de movimiento y gaussiano, ruido de sensor y compresión JPEG;
  - reflejos especulares, sombras, poca luz (gamma) y balance de blancos.
- **REQ-B003-08**: Cada imagen sintética DEBERÁ llevar etiquetas YOLO **exactas** de cada glifo
  y metadatos (layout, valores y parámetros de render), y el lote DEBERÁ ser reproducible con su
  semilla.

### Datos reales
- **REQ-B003-09**: DEBERÁ existir un protocolo de captura (design §4) que se pueda seguir sin
  conocimientos técnicos. La etiqueta de la lectura se codifica en el nombre del fichero.
- **REQ-B003-10**: `ingest` DEBERÁ:
  - validar el nombre del fichero;
  - extraer frames de los vídeos (por defecto 3 fps);
  - descartar los borrosos (varianza del laplaciano por debajo del umbral);
  - deduplicar (hash perceptual);
  - recortar la ROI simulando la de la app (centrada, el 80 % del ancho y una relación de aspecto
    de 4:5);
  - generar un manifiesto JSONL con el sha256 de cada fichero.
- **REQ-B003-11**: Los datos reales NO DEBERÁN versionarse en git (C-ML-03). Solo el manifiesto.
  Su ubicación se configura con `TENSOSCAN_ML_DATA_DIR`.
- **REQ-B003-12**: `split` DEBERÁ separar por **modelo de tensiómetro** (agrupado): al menos un
  modelo completo DEBERÁ quedar solo en test.
- **REQ-B003-13**: `prelabel` DEBERÁ generar pre-anotaciones (formato CVAT/YOLO) con el mejor
  modelo disponible para el subconjunto real que se anote con cajas.

### Modelo, entrenamiento y evaluación
- **REQ-B003-14**: La arquitectura DEBERÁ elegirse mediante un *spike* documentado (ADR-B003-02)
  que cumpla: licencia compatible con la de la app, exportación a LiteRT solo con *builtins* y
  presupuesto del contrato §7.
- **REQ-B003-15**: El entrenamiento DEBERÁ hacerse en dos etapas: preentrenamiento sintético y
  *fine-tuning* mixto (real con cajas + sintético). Se guardarán los *checkpoints* y las curvas.
- **REQ-B003-16**: `eval` DEBERÁ calcular, a partir de las detecciones de **cualquier** detector
  (incluido el `.tflite` exportado), las métricas de §1 con desglose por dispositivo, por
  condición y por clase (matriz de confusión de glifos), y generar `report.md` y `report.json`.
- **REQ-B003-17**: `eval` DEBERÁ evaluar el `.tflite` exportado (no solo el *checkpoint*) con el
  mismo pre y postprocesado que la app (letterbox, umbral, NMS agnóstico, deshacer letterbox).

### Exportación y publicación
- **REQ-B003-18**: `export` DEBERÁ producir `digits-<semver>.tflite` (float16) y, opcionalmente,
  `-int8` (cuantización con un dataset representativo de 300 imágenes), cuya salida cumpla el
  contrato §4 (decodificación y sigmoide dentro del grafo, sin NMS).
- **REQ-B003-19**: `verify` DEBERÁ comprobar con el runtime LiteRT de Python:
  - las formas y tipos de E/S;
  - que todas las operaciones son *builtins*;
  - el número de clases y su orden;
  - el tamaño;
  - la latencia en CPU (indicativa);
  - que el modelo reproduce las lecturas de un set de 20 imágenes dorado.
- **REQ-B003-20**: `release` DEBERÁ generar el sidecar JSON (contrato §5) con el sha256 y las
  métricas reales, el *model card* y el paquete de *release*. Solo DEBERÁ permitirlo SI se cumplen
  los umbrales de §1 (C-ML-04).
- **REQ-B003-21**: La entrega a la app DEBERÁ hacerse con un PR a `TensoScan_App` que añada los
  artefactos a `data/src/main/assets/models/` y actualice `DIGIT_DETECTOR_PRIMARY` (tarea App
  T-A002-35).

### CI
- **REQ-B003-22**: En CI DEBERÁN ejecutarse los tests unitarios de `ml/` (parser, validador,
  geometría del render, etiquetas, ingest con medios sintéticos y decodificador). El
  entrenamiento real NO se ejecuta en CI.

## 4. Criterios de aceptación

| ID | Escenario | Req. |
|----|-----------|------|
| CA-B003-01 | Dada la semilla 42, cuando genero 100 imágenes dos veces, entonces los ficheros son idénticos byte a byte | 02, 08 |
| CA-B003-02 | Dada una imagen sintética, cuando dibujo sus etiquetas, entonces cada caja ciñe su glifo (comprobado por test geométrico: IoU ≥ 0,9 frente a la máscara renderizada) | 08 |
| CA-B003-03 | Dados los casos dorados, cuando ejecuto los tests de `tensoscan_ml.reading`, entonces pasan todos | 04 |
| CA-B003-04 | Dado `omron-m3__128-82-67__dim__01.mp4`, cuando lo ingesto, entonces obtengo frames recortados con la etiqueta 128/82/67, sin borrosos ni duplicados, y entradas en el manifiesto | 10 |
| CA-B003-05 | Dado un `.tflite` exportado, cuando ejecuto `verify`, entonces todas las comprobaciones pasan | 19 |
| CA-B003-06 | Dadas métricas por debajo del umbral, cuando ejecuto `release`, entonces falla con el detalle | 20 |

## 5. Fuera de alcance
- Detector de esquinas del display (v2, según App ADR-A002-04).
- Descarga remota y OTA de modelos.
- Entrenamiento con imágenes de usuarios de producción (requiere una spec de consentimiento).

## 6. Preguntas abiertas

| ID | Pregunta | Propuesta |
|----|----------|-----------|
| Q-B003-01 | ¿La app será comercial o de código cerrado? (Afecta a las licencias AGPL: Ultralytics) | Usar arquitecturas Apache-2.0 por defecto |
| Q-B003-02 | ¿Qué tensiómetros reales hay disponibles? (≥ 3 modelos; ideal ≥ 5 de marcas distintas) | Necesita respuesta |
| Q-B003-03 | ¿Hay GPU disponible (local o Colab/Kaggle) para entrenar? | Colab o Kaggle (gratis) para v1 |
| Q-B003-04 | ¿Dónde se guardan los datos reales (disco local, Drive o bucket)? | Carpeta local + copia en Drive; solo el manifiesto en git |
