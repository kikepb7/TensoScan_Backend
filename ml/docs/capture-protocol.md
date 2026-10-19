# Protocolo de captura de datos reales

Objetivo: reunir imágenes del display de tensiómetros con la lectura correcta conocida, para (1) entrenar y (2) **evaluar
la aceptación** del modelo (spec B-003). No hace falta dibujar cajas: la lectura va en el nombre del fichero o en un
`labels.csv`. Las imágenes **no se suben al repositorio** (solo el manifiesto con hashes).

## 1. Qué necesitas

- El tensiómetro (cuantos más modelos o marcas distintos, mejor) y el móvil con la cámara normal.
- Una carpeta para guardar el material (disco local o Drive). Su ruta será `TENSOSCAN_ML_DATA_DIR`.

## 2. Vídeos (recomendado cuando puedes medir de nuevo)

Por cada medición real:

1. Haz la medición con normalidad y **anota los tres valores** (SYS, DIA, PUL) que muestra el display.
2. Graba un **vídeo de 15 a 30 s** apuntando al display y moviendo el móvil despacio: acércalo y aléjalo, inclínalo un
   poco a cada lado. No hace falta que salga perfecto: los defectos también sirven.
3. Repite en 2 o 3 condiciones de luz cuando puedas:
   - `good`: luz natural o de lámpara;
   - `dim`: poca luz;
   - `glare`: con un reflejo de lámpara o ventana sobre la pantalla.
4. Nombra el fichero así:

   ```
   <marca>-<modelo>__<sys>-<dia>-<pul>__<condición>__<nn>.mp4
   ejemplo: omron-m3__128-82-67__dim__01.mp4
   ```

   - marca y modelo en minúsculas, sin espacios (`-` como separador);
   - si el display muestra un error, escribe `error` en lugar de los valores: `omron-m3__error__good__01.mp4`;
   - `nn` es el número de toma (`01`, `02`...).
5. Encuadra **solo el display**, sin la persona. Consentimiento: quien aparezca o cuyos valores se registren debe
   consentirlo.

Objetivo mínimo para la v1 (`production`): **3 modelos o más** de tensiómetro × **10 lecturas distintas o más** ×
**2 condiciones**, unos 60 vídeos. Para el canal `beta` basta 1 dispositivo con 30 lecturas distintas o más.

## 3. Fotos ya existentes (con `labels.csv`)

No hace falta renombrarlas. Crea un `labels.csv` en la misma carpeta que las fotos:

```csv
file,device,sys,dia,pul,condition
IMG_0001.jpg,omron-m3,128,82,67,good
IMG_0002.jpg,omron-m3,141,94,78,glare
IMG_0003.jpg,omron-m3,error,,,dim
```

- `file`: ruta de la foto relativa a la carpeta del `labels.csv`.
- `sys`, `dia`, `pul`: enteros; en un error escribe `error` en `sys` y deja `dia` y `pul` vacíos.
- `condition`: `good`, `dim` o `glare`.
- Las lecturas se validan con las reglas compartidas (`reading-rules.md`): una lectura imposible se rechaza y se informa.

## 4. Encuadre (`--roi-mode`)

La app analiza un recorte centrado (80 % del ancho, proporción 4:5). Hay que parecerse a él:

| Situación | Opción | Qué hace |
|-----------|--------|----------|
| Fotos/vídeos con el display ocupando la zona central | `center` (por defecto) | Aplica el mismo recorte que la app |
| La imagen ya está recortada sobre el display | `none` | Usa la imagen entera |
| Fotos del aparato entero | `manual` | Usa `roi_x,roi_y,roi_w,roi_h` (píxeles) de cada fila de `labels.csv` |

Antes de procesar un lote, **comprueba el encuadre** con la hoja de contactos:

```bash
export TENSOSCAN_ML_DATA_DIR=/ruta/a/los/datos
uv run tensoscan-ml ingest --src /ruta/a/las/fotos --roi-mode center --preview
# abre $TENSOSCAN_ML_DATA_DIR/real/<dataset_version>/preview.jpg: a la izquierda la imagen con el recorte
# marcado en verde, a la derecha el recorte. Si no encuadra el display, usa `none` o `manual`.
```

Ningún fichero se descarta en silencio: `ingest_report.json` lista cada rechazo (nombre incorrecto, fila inválida,
fichero que no existe, borroso, duplicado) con su motivo.

## 5. Captura para la evaluación de aceptación

El modelo solo se publica si cumple las métricas **en datos reales que no ha visto** (C-ML-04). Por eso:

1. Reserva material que **no** se use para entrenar. `split` lo hace por ti: con 3 o más dispositivos deja uno entero en
   test; con menos agrupa por **lectura**, de modo que ninguna combinación SYS/DIA/PUL aparezca en entrenamiento y test.
2. Para que la prueba sea honesta, incluye lecturas **distintas** entre sí (no repitas la misma medición muchas veces) y
   al menos una toma con `dim` y otra con `glare`.
3. Con un solo dispositivo la evaluación es `beta`: sus cifras son optimistas porque el aparato ya se ha visto. Para
   `production` hacen falta 3 dispositivos de marcas o tipos de LCD distintos, con uno excluido del entrenamiento.
4. Incluye también algunas tomas de `error` (`E1`, `Err`) y de «midiendo» (`---`) si el aparato las muestra.

## 6. Entrega

```bash
uv run tensoscan-ml ingest --src /ruta/a/las/capturas --dataset-version ds-real-0.1.0 --commit-manifest
uv run tensoscan-ml split --manifest $TENSOSCAN_ML_DATA_DIR/real/ds-real-0.1.0/manifest.jsonl
```

`--commit-manifest` copia el manifiesto (sin imágenes) a `ml/datasets/manifests/`, que es lo único que se versiona en git.
