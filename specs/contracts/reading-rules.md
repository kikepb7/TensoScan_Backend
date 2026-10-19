# Reglas de lectura compartidas (validación y parseo de layout)

> **Fuente canónica**: `TensoScan_Backend/specs/contracts/reading-rules.md` y
> `reading-golden-cases.json`. Copias sincronizadas en `TensoScan_App/specs/contracts/`.
> Consumidores:
> - App: `ReadingValidator` y `DisplayLayoutParser` (`:domain`, spec A-002).
> - Backend: validación de `POST /v1/measurements` (spec B-002) y harness de evaluación del
>   modelo (spec B-003), que **reutiliza el mismo parser** para medir el acierto por lectura.
>
> Versión: **1.0.0** · Estado: 📝 Pendiente de aprobación

## 1. Por qué existe este contrato

Las mismas reglas se aplican en tres sitios: al aceptar una lectura en la cámara, al guardarla en
el servidor y al medir la calidad del modelo. Si divergen aparecen errores silenciosos: la app
acepta valores que el servidor rechaza, o la métrica de evaluación no refleja lo que verá el
usuario. Por eso las reglas se escriben **una sola vez**, con casos dorados ejecutables
(`reading-golden-cases.json`) que ambos repositorios ejecutan en sus tests.

## 2. Rangos absolutos (errores bloqueantes)

| Campo | Unidad | Mínimo | Máximo | Motivo |
|-------|--------|--------|--------|--------|
| `systolic` | mmHg | 60 | 260 | Fuera de este rango la lectura es casi con seguridad un error de reconocimiento o de tecleo; los tensiómetros domésticos miden hasta ~280–300 mmHg, pero valores > 260 en domicilio son excepcionales |
| `diastolic` | mmHg | 30 | 160 | Ídem |
| `pulse` | ppm | 30 | 220 | Rango que cubren los tensiómetros domésticos (≈ 40–180) con margen |

Los límites son **inclusivos**. Todos los valores son enteros.

## 3. Validación

### 3.1 Errores (impiden guardar)

Se evalúan en este orden y se devuelven **todos** los que apliquen:

| Código | Campo | Regla |
|--------|-------|-------|
| `REQUIRED` | cualquiera | El valor falta |
| `OUT_OF_RANGE` | cualquiera | Fuera de §2 |
| `SYSTOLIC_NOT_GREATER_THAN_DIASTOLIC` | `systolic` | Solo si SYS y DIA no tienen error propio: `systolic <= diastolic` |
| `IN_FUTURE` | `measured_at` | `measured_at > now + 5 min` (solo al guardar) |
| `TOO_OLD` | `measured_at` | `measured_at < 2000-01-01T00:00:00Z` (solo al guardar) |

### 3.2 Avisos (permiten guardar previa confirmación explícita)

Se evalúan **solo si no hay errores**, en este orden:

| Código | Regla | Mensaje orientativo (la app lo localiza) |
|--------|-------|------------------------------------------|
| `VERY_HIGH_PRESSURE` | `systolic >= 180` **o** `diastolic >= 120` | «Valores muy elevados. Repite la medición tras 5 minutos de reposo. Si se mantienen o tienes síntomas, busca atención médica urgente.» |
| `LOW_PRESSURE` | `systolic < 90` **o** `diastolic < 60` | «Valores bajos. Comprueba que la lectura es correcta.» |
| `UNUSUAL_PULSE` | `pulse < 40` **o** `pulse > 150` | «Pulso inusual. Comprueba la lectura.» |
| `UNUSUAL_PULSE_PRESSURE` | `systolic - diastolic < 20` **o** `> 100` | «La diferencia entre sistólica y diastólica es inusual. Comprueba la lectura.» |
| `POSSIBLE_DUPLICATE` | (solo app) existe otra medición con los mismos tres valores y `|Δ measured_at| <= 5 min` | «Ya guardaste una medición idéntica hace un momento.» |

Los umbrales de `VERY_HIGH_PRESSURE` siguen el criterio de «crisis hipertensiva» de las guías
AHA/ACC y ESC/ESH (≥ 180 y/o ≥ 120). Los textos son orientativos y **deben revisarlos un
profesional sanitario** antes de publicar (pregunta abierta Q-SAFE-01 en A-002).

## 4. Parser de layout del display (de detecciones a SYS/DIA/PUL)

### 4.1 Entrada

Lista de detecciones `{symbol, box: [left, top, right, bottom], confidence}`:
- `symbol` ∈ `0`…`9`, `colon`, `dash`, `E` y `r` (clases del modelo, ver
  `digit-detector-model.md`);
- `box` normalizada a la ROI, `[0, 1]`, con el origen arriba a la izquierda;
- `confidence` ∈ `[0, 1]`.

### 4.2 Constantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `MIN_SYMBOL_CONFIDENCE` | 0.40 | Las detecciones por debajo se descartan |
| `MIN_SYMBOL_HEIGHT` | 0.04 | Las cajas más bajas (en proporción de la altura de la ROI) se descartan como ruido |
| `GROUP_MIN_VERTICAL_OVERLAP` | 0.60 | Solape vertical mínimo (sobre la altura menor) para unir dos glifos en un número |
| `GROUP_MAX_HEIGHT_RATIO` | 1.35 | Relación máxima de alturas para unirlos |
| `GROUP_MAX_GAP_FACTOR` | 0.35 | Hueco horizontal máximo, en múltiplos de la altura mayor |
| `GROUP_MAX_OVERLAP_FACTOR` | 0.30 | Solape horizontal máximo tolerado, en múltiplos de la anchura menor |
| `ROW_MIN_VERTICAL_OVERLAP` | 0.50 | Solape vertical mínimo para que dos números estén en la misma fila |
| `PRIMARY_MIN_HEIGHT_RATIO` | 0.75 | Altura mínima (frente al número más alto) para ser candidato a SYS o DIA |
| `SECONDARY_MIN_HEIGHT_RATIO` | 0.35 | Altura mínima para ser candidato a PUL; los números más pequeños (fecha, memoria) se ignoran |

Las constantes están afinadas con los casos dorados. Cambiarlas implica una nueva versión del
contrato y regenerar los casos.

### 4.3 Algoritmo (determinista)

Notación: `h(b) = bottom − top`, `w(b) = right − left`, y
`vov(a, b) = max(0, min(a.bottom, b.bottom) − max(a.top, b.top)) / min(h(a), h(b))`.

1. **Filtrar** las detecciones con `confidence < MIN_SYMBOL_CONFIDENCE` o `h(box) < MIN_SYMBOL_HEIGHT`.
2. **Agrupar en números**: ordenar por `(left, top)` y recorrer. Cada detección `d` se añade al
   grupo existente cuyo **último** elemento `l` cumpla todas estas condiciones:
   - `vov(l, d) >= GROUP_MIN_VERTICAL_OVERLAP`;
   - `max(h)/min(h) <= GROUP_MAX_HEIGHT_RATIO`;
   - `gap = d.left − l.right` dentro de `[−GROUP_MAX_OVERLAP_FACTOR·min(w), GROUP_MAX_GAP_FACTOR·max(h)]`.

   Si hay varios grupos válidos, se elige el de **menor `gap`**. Si no hay ninguno, `d` abre un
   grupo nuevo.
3. **Clasificar cada grupo**:
   - si contiene `E` → **grupo de error**;
   - si son ≥ 2 `dash` y nada más → **grupo «midiendo»**;
   - si contiene `colon` → **hora** (se ignora);
   - si son solo dígitos y tiene 2 o 3 → **número** (su valor es la concatenación, su caja es la
     envolvente y su confianza es la media);
   - en cualquier otro caso → se ignora (dígitos sueltos como el índice de memoria, ≥ 4 dígitos,
     mezclas).
4. **Estado del frame**:
   - si existe algún grupo de error → `DEVICE_ERROR` (los campos son nulos);
   - si no hay números → `MEASURING` si existe un grupo «midiendo» y `NO_READING` en caso
     contrario.
5. **Orden de lectura**: con `H` = altura del número más alto, los *secundarios* son los números
   con `h >= SECONDARY_MIN_HEIGHT_RATIO·H`. Se agrupan en **filas**: se recorren ordenados por su
   centro vertical y cada uno entra en la primera fila existente cuyo **primer** elemento cumpla
   `vov >= ROW_MIN_VERTICAL_OVERLAP`, o abre una fila nueva. Las filas se ordenan por su `top`
   mínimo y, dentro de cada fila, los números se ordenan por `left`. El resultado es la lista de
   lectura.
6. **Asignación**:
   - *primarios* = elementos de la lista de lectura con `h >= PRIMARY_MIN_HEIGHT_RATIO·H`;
   - si hay menos de 2 primarios → `NO_READING`;
   - `SYS` = primer primario y `DIA` = segundo primario;
   - `PUL` = primer elemento de la lista de lectura **posterior a DIA** (distinto de SYS);
   - estado `COMPLETE` si hay PUL y `PARTIAL` si no lo hay.
7. **Sin correcciones mágicas**: el parser **no** intercambia SYS y DIA ni completa dígitos. Si
   `SYS <= DIA`, la validación (§3) lo rechaza y la app sigue escaneando.

### 4.4 Salida

```
{ status: COMPLETE | PARTIAL | NO_READING | MEASURING | DEVICE_ERROR,
  systolic: int?, diastolic: int?, pulse: int?,
  confidences: { systolic: float?, diastolic: float?, pulse: float? } }
```
Los casos dorados comprueban `status`, `systolic`, `diastolic` y `pulse`. La confianza de cada
campo es la media de la confianza de sus glifos.

### 4.5 Layouts cubiertos y límites conocidos

| Cubierto | No cubierto en v1 (se resuelve en la pantalla de revisión) |
|----------|-------------------------------------------------------------|
| SYS, DIA y PUL apilados (PUL pequeño o del mismo tamaño) | Displays con SYS y DIA de distinto tamaño (< 75 %) |
| SYS \| DIA en la misma fila con PUL debajo | Pulso mostrado **antes** que DIA en el orden de lectura |
| Hora (`hh:mm`) y fecha pequeña en pantalla | Fecha grande sin separador (≥ 35 % de la altura) junto a los valores |
| Índice de memoria de un dígito | Tensiómetros de muñeca con display rotado 90° |
| Códigos `E1` y `Err`, y `---` | Displays de matriz de puntos o LCD con fuentes no de 7 segmentos |

## 5. Casos dorados

`reading-golden-cases.json` contiene:
- `parser_cases[]`: `{ name, description, detections[], expected{status, systolic, diastolic, pulse} }`;
- `validation_cases[]`: `{ name, input{systolic, diastolic, pulse}, errors[{field, code}], warnings[] }`.

Reglas:
- Ambos repositorios ejecutan **todos** los casos en su suite unitaria: app con `@TestFactory` y
  backend con `pytest.mark.parametrize`.
- Un caso nuevo se añade aquí primero y luego se copia. Los casos no se borran: se marcan con
  `"deprecated": true` y una nueva versión del contrato.
- Los casos se generaron y verificaron con una implementación de referencia del algoritmo §4.3.
  La implementación oficial en Python vivirá en `ml/tensoscan_ml/reading/parser.py` (B-003).

## 6. Historial

| Versión | Fecha | Cambio |
|---------|-------|--------|
| 1.0.0 | 2026-10-19 | Versión inicial: 14 casos de parser y 14 de validación |
