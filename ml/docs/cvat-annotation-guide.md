# Guía de anotación de glifos en CVAT

Objetivo: dibujar una caja alrededor de **cada glifo de 7 segmentos encendido** en unos 300 fotogramas reales. Las cajas
sirven para el *fine-tuning* y para medir el mAP real. No hace falta anotar todo el material: la lectura correcta ya va
en el nombre del fichero o en `labels.csv`.

Un modelo hace el primer borrador (`tensoscan-ml prelabel`) y tú solo corriges. Tiempo estimado: 20–30 s por imagen.

## 1. Generar el borrador

```bash
export TENSOSCAN_ML_DATA_DIR=/ruta/a/los/datos
# con el mejor modelo disponible (un checkpoint o un .tflite); los fotogramas salen de `ingest`
uv run tensoscan-ml prelabel \
  --manifest $TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0/manifest.jsonl \
  --detector tflite --model dist/digits-0.1.0.tflite \
  --out $TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0/annotate --count 300
```

Genera en `annotate/`:

| Fichero | Para qué |
|---------|----------|
| `images/` | los ~300 fotogramas elegidos (repartidos por dispositivo, condición y vídeo) |
| `labels/` | las cajas propuestas (formato YOLO) |
| `cvat_yolo11.zip` | las mismas cajas en formato «YOLO 1.1» para importarlas en CVAT |
| `confidences.csv` | imágenes ordenadas de menos a más confianza: revisa primero las primeras |
| `selected.jsonl` | manifiesto de los fotogramas elegidos, con la ruta de sus etiquetas |

Con un modelo `dummy` el borrador no sirve (es ruido): en ese caso importa solo las imágenes y dibuja las cajas desde cero, o
espera al primer modelo entrenado con sintético (suele dar un borrador aprovechable).

## 2. Crear el proyecto y la tarea en CVAT

1. Entra en <https://app.cvat.ai> (o en tu CVAT local con Docker) y crea un **proyecto** `tensoscan-digits`.
2. Añade **14 etiquetas de tipo Rectangle, con estos nombres exactos y en este orden**:
   `0 1 2 3 4 5 6 7 8 9 colon dash E r`
   (`colon` = los dos puntos de la hora, `dash` = el guion `-`, `E` y `r` = los códigos `E1`, `Err`).
3. Crea una **tarea** dentro del proyecto y sube la carpeta `images/`.
4. Abre la tarea → menú **Actions → Upload annotations** → formato **YOLO 1.1** → elige `cvat_yolo11.zip`.

## 3. Reglas de anotación

- **Una caja por glifo**, ajustada a los segmentos **encendidos** (con un margen mínimo, como máximo un par de píxeles).
  No incluyas los segmentos apagados («fantasma»): la caja no cubre el `8` pálido de fondo.
- El **`1`** es estrecho: su caja también. No la ensanches hasta la celda del dígito.
- Anota **todos** los glifos de 7 segmentos visibles, también los pequeños: la hora (`hh:mm`, con `colon`), la fecha, el
  índice de memoria y los guiones `---` (`dash`). El parser decide después cuáles cuentan.
- **No** anotes textos ni iconos que no son 7 segmentos: `SYS`, `DIA`, `PUL`, `mmHg`, el corazón, la batería, la `M`.
- Códigos de error: `E` y cada `r` por separado (`Err` son tres cajas: `E`, `r`, `r`).
- Un glifo cortado por el borde de la imagen: anótalo solo si se ve **más de la mitad**; la caja llega hasta el borde.
- Un glifo tapado por un reflejo: anótalo si **se puede adivinar con seguridad** cuál es; si no, no lo anotes (es mejor una
  ausencia que una clase dudosa).
- Cajas que se solapan: no debería ocurrir entre glifos distintos. Si el borrador dibuja dos cajas sobre el mismo glifo,
  borra la sobrante.

## 4. Qué corregir en el borrador

1. **Falsos positivos**: cajas sobre manchas, reflejos o texto → bórralas.
2. **Clase equivocada**: `5` por `6`, `1` por `7`, `0` por `8`... → cambia la etiqueta.
3. **Glifos que faltan** → dibuja la caja (atajo `N` para una caja nueva).
4. **Cajas flojas o apretadas** → ajústalas.
5. Cuando termines, comprueba que el número de cajas de cada fila coincide con la lectura conocida.

## 5. Exportar y comprobar

1. **Actions → Export task dataset** → formato **YOLO 1.1** (sin imágenes).
2. Descomprime y copia los ficheros `obj_train_data/*.txt` sobre `annotate/labels/` **conservando el nombre**.
3. Comprueba las anotaciones con el parser oficial:

   ```bash
   uv run tensoscan-ml prelabel --check $TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0/annotate/selected.jsonl
   ```

   Compara las cajas de cada fotograma con la lectura conocida (`labels.csv` o el nombre del fichero) y lista las
   imágenes que no la reproducen, con el motivo. Un fallo no siempre es un error de anotación (con fuentes anchas el
   parser compartido no une un `1` a su vecino izquierdo: hallazgo F1 del `README`), pero es la forma más rápida de
   encontrar un glifo olvidado o con la clase cambiada.
4. Reparte los fotogramas en train/val/test sin fugas y úsalos en el *fine-tuning*:

   ```bash
   uv run tensoscan-ml split --manifest $TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0/annotate/selected.jsonl
   # y en configs/train/yolox-nano-v1.yaml: data.real.train -> .../annotate/train.jsonl, data.real.val -> .../annotate/val.jsonl
   ```

Las imágenes, las etiquetas y los zips **no se suben a git**; solo el manifiesto (`ingest --commit-manifest`).
