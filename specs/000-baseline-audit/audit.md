# 000 — Auditoría del estado actual (Backend)

> Estado: ✅ Redactada · Rama base analizada: `master` @ `e345cda`
> Severidad: 🔴 Crítico · 🟠 Alto · 🟡 Medio · ⚪ Bajo. Rutas relativas a la raíz del repo.

## 1. Resumen ejecutivo

1. **El reconocimiento no es fiable por diseño, no solo por el modelo.** Recorta los dígitos en
   **coordenadas fijas** tras redimensionar a 1109×1431, asume siempre 3+2+2 dígitos, no corrige
   la perspectiva, corrompe la imagen en el preprocesado (desbordamiento de `uint8`) y clasifica
   con un modelo de tipo MNIST (28×28) que no sirve para 7 segmentos. Se sustituye por
   reconocimiento en el dispositivo (App A-002) y un modelo nuevo (B-003).
2. **Fragilidad operativa grave.** El modelo Keras se carga **al importar** el router. Si el
   fichero no existe (no está en el repo), **toda la API cae**, incluido el login. Además se
   escriben ficheros temporales con nombre fijo en el directorio de trabajo, lo que provoca
   condiciones de carrera entre peticiones, y existe código con `cv2.imshow`/`waitKey(0)`.
3. **Seguridad**: `SECRET_KEY` por defecto `"supersecret"`, `.env` copiado dentro de la imagen
   Docker, Mongo sin autenticación expuesto en el host, `python-jose` (con CVEs conocidas), sin
   rate limit y con errores internos devueltos al cliente (`detail=str(e)`).
4. **Calidad**: sin tests, sin linters, sin CI (la carpeta `.github/workflow` está mal nombrada),
   dependencias sin depurar (TensorFlow, Plotly, Matplotlib, 5 librerías de PDF) y Python 3.9
   (fin de soporte en octubre de 2025).

## 2. Hallazgos

### 2.1 Reconocimiento (servidor)

| ID | Sev. | Ubicación | Hallazgo | Resuelve |
|----|------|-----------|----------|----------|
| AUD-B-01 | 🔴 | `app/infrastructure/services/image_processor_service.py:108-116` | Dígitos recortados en **coordenadas fijas**: solo vale para un modelo de tensiómetro y un encuadre exacto | App A-002 + B-002 (retirada) |
| AUD-B-02 | 🔴 | `app/infrastructure/services/display_recognizer_service.py:62-64` | Supone 3+2+2 dígitos: una SYS < 100 o un PUL ≥ 100 se trocean mal; las posiciones apagadas se clasifican igualmente como un dígito | Ídem (parser de `reading-rules.md`) |
| AUD-B-03 | 🟠 | `image_processor_service.py:89` | `boundingRect` sin corrección de perspectiva; el contorno más grande de 4 vértices suele ser la carcasa | Ídem |
| AUD-B-04 | 🔴 | `app/infrastructure/services/keras_number_recognizer.py:23` | `img * 255` sobre `uint8` desborda y corrompe la entrada | Ídem |
| AUD-B-05 | 🟠 | `keras_number_recognizer.py:25-26` | `bitwise_not` + `THRESH_BINARY_INV` con `maxval=120`: doble inversión y rango 0/120 | Ídem |
| AUD-B-06 | 🟠 | `keras_number_recognizer.py` | Clasificador 28×28 de tipo MNIST para glifos de 7 segmentos, sin clase «vacío» | B-003 |
| AUD-B-07 | 🔴 | `display_recognizer_service.py:19`, `image_processor_service.py:127` | Escribe `display.jpg` y `digit_region_*.png` en el CWD con nombre fijo: carreras entre peticiones y datos de salud en disco | B-001 T-B001-03 (aislamiento) · B-002 T-B002-09 (retirada) |
| AUD-B-08 | 🟠 | `image_processor_service.py:179-180` | `cv2.imshow` + `waitKey(0)`: bloquearía el worker si se invocara | Ídem |
| AUD-B-09 | 🔴 | `app/api/routes/ocr_routes.py:23-26` | Modelo cargado **al importar**, con una ruta relativa frágil (`../ia_models/tensoscan_model.keras`) a un fichero que no está en el repo. Si falta, no arranca **ninguna** ruta | T-B001-03 |
| AUD-B-10 | 🟠 | `ocr_routes.py:37-48` | `/ocr/recognize` sin autenticación: inferencia gratuita para cualquiera (DoS) | T-B001-03 (protegido) · T-B002-09 (retirada) |
| AUD-B-11 | 🟠 | `ocr_routes.py:51-73` | Guarda el resultado **sin confirmación** del usuario y sin límite de tamaño de subida | B-002 (las mediciones solo llegan confirmadas) |

### 2.2 API, arquitectura y errores

| ID | Sev. | Ubicación | Hallazgo | Resuelve |
|----|------|-----------|----------|----------|
| AUD-B-12 | 🟠 | `ocr_routes.py:118,139,159,171` + `except Exception` en `:124,151,173` | Los `HTTPException` 400 y 404 se capturan y se convierten en 500 | T-B001-04 |
| AUD-B-13 | 🟠 | `ocr_routes.py:125,152` | `detail=str(e)`: filtra detalles internos al cliente | T-B001-04 |
| AUD-B-14 | 🟠 | `ocr_routes.py`, `auth_routes.py` | Los routers acceden directamente a las colecciones Mongo y contienen lógica de negocio | T-B001-05 |
| AUD-B-15 | 🟠 | `app/domain/models/display_result_model.py` | Valores de medición como `str` y `confidence` sin rango; sin validación fisiológica | B-002 |
| AUD-B-16 | 🟡 | `ocr_routes.py:66` | `timestamp = datetime.now()` (naive, hora local del servidor) y sin distinguir la hora de la medición de la de creación | B-002 |
| AUD-B-17 | 🟡 | `ocr_routes.py:83-84` | `to_list(length=None)` sin paginación ni orden | B-002 |
| AUD-B-18 | 🟡 | Rutas | Prefijos incoherentes (`/ocr/user/measurements/pdf` frente a `/ocr/measurements`), sin versionado | B-002 (`/v1`) |
| AUD-B-19 | 🟡 | `app/main.py:16-25` | La raíz anuncia endpoints inexistentes (`/ocr/test`...); el título es «Number Recognition API» | T-B001-03 |
| AUD-B-20 | 🟡 | `app/infrastructure/services/get_user_service.py:9` | `tokenUrl="/auth/login"`, pero la ruta real es `/login`: Swagger no puede autenticar | T-B001-07 |
| AUD-B-21 | 🟠 | `app/utils/html_render.py:22-30,112-115` | Valores interpolados sin escapar en HTML y en JS (`var fechas = {fechas}` con el `repr` de Python): inyección | B-002 (se elimina el HTML) |
| AUD-B-22 | ⚪ | `html_render.py:3` | `from pygments.lexers import go`: import sin sentido | B-002 |
| AUD-B-23 | 🟡 | `html_render.py:36` | Plotly `latest` desde CDN: no reproducible; además el PDF (xhtml2pdf) no ejecuta JS, así que el gráfico nunca aparece en el PDF | B-002 (PDF con ReportLab) |
| AUD-B-24 | 🔴 | `app/domain/models/chatbot_model.py:2` | `from openai import BaseModel`: depende de un detalle interno del SDK de OpenAI para los modelos de dominio | T-B001-09 |
| AUD-B-25 | 🟡 | `app/api/routes/chatbot_routes.py:15-23,31` | Bloque `try` de inicialización vacío, `load_dotenv` duplicado, `os.getenv` directo, modelo `gpt-3.5-turbo` fijo, sin *system prompt* de seguridad clínica y con historial ilimitado | T-B001-09 |
| AUD-B-26 | ⚪ | `auth_routes.py:21` | Respuesta `{"msg:": ...}` (clave con dos puntos) y 200 en lugar de 201 | T-B001-07 |

### 2.3 Seguridad

| ID | Sev. | Ubicación | Hallazgo | Resuelve |
|----|------|-----------|----------|----------|
| AUD-B-27 | 🔴 | `app/core/config.py:22` | `SECRET_KEY` por defecto `"supersecret"`: cualquiera puede forjar tokens si no se configura | T-B001-02 |
| AUD-B-28 | 🔴 | `Dockerfile:14`, `.dockerignore` | `COPY . .` sin excluir `.env`: los secretos quedan dentro de la imagen | T-B001-10 |
| AUD-B-29 | 🔴 | `docker-compose.yaml:11-16` | Mongo sin autenticación y con `27017` publicado en el host; datos en `./data` sin ignorar en git | T-B001-10 |
| AUD-B-30 | 🟠 | `requirements.txt` (`python-jose` sin versión fijada) | `python-jose` tiene vulnerabilidades conocidas y poco mantenimiento | T-B001-06 (PyJWT) |
| AUD-B-31 | 🟡 | `app/infrastructure/services/auth_service.py:8` | `passlib` sin mantenimiento e incompatibilidades conocidas con `bcrypt >= 4.1` | T-B001-06 |
| AUD-B-32 | 🟡 | `auth_service.py:20` | `exp` calculado con `datetime.now()` naive; se ignora `ACCESS_TOKEN_EXPIRE_MINUTES` (expira a los 15 min) | T-B001-06 |
| AUD-B-33 | 🟠 | `auth_routes.py` | Sin rate limit en login ni registro (fuerza bruta) | T-B001-08 |
| AUD-B-34 | 🟡 | `auth_routes.py:12-19` | Unicidad de email comprobada solo en la aplicación (carrera), sin índice único y sin normalizar mayúsculas | T-B001-07 |
| AUD-B-35 | 🟡 | `get_user_service.py:19` | `ObjectId(user_id)` sin validar → 500 con un token manipulado | T-B001-06 |
| AUD-B-36 | 🟡 | — | Sin refresh tokens: la app pierde la sesión cada 15 minutos | T-B001-07 |

### 2.4 Operación, dependencias y calidad

| ID | Sev. | Ubicación | Hallazgo | Resuelve |
|----|------|-----------|----------|----------|
| AUD-B-37 | 🟠 | `Dockerfile:2,5-8` | `python:3.9-slim` (EOL). `libgl1-mesa-glx` ya no existe en Debian actual (el build falla). Se ejecuta como root y sin `HEALTHCHECK` | T-B001-10 |
| AUD-B-38 | 🟠 | `requirements.txt` | ~120 dependencias fijadas a mano: TensorFlow, Plotly, Kaleido, Matplotlib, 5 librerías de PDF, `pip` y `setuptools` fijados, `motor 2.5` + `pymongo 3.12` antiguos (Motor está deprecado en favor de la API async de PyMongo) | T-B001-01 |
| AUD-B-39 | 🟠 | Repo | Sin tests, sin `ruff`/`mypy` y sin CI | T-B001-01, 11 |
| AUD-B-40 | ⚪ | `.github/workflow/pull_request_template.md` | Ruta incorrecta: GitHub no reconoce la plantilla | T-B001-11 |
| AUD-B-41 | 🟡 | `*.py` | `print` en lugar de logging (`image_processor_service.py:96,128-129`, `pdf_generator.py:12`); `logging.basicConfig` en cada router | T-B001-03 |
| AUD-B-42 | 🟡 | `app/infrastructure/database/mongo_database.py` | Cliente creado al importar, sin cierre, sin índices y sin `lifespan` | T-B001-03, 05 |
| AUD-B-43 | 🟡 | `app/core/config.py` | Mezcla `load_dotenv` + `os.getenv` dentro de `BaseSettings` (anula su propósito) y usa `class Config` de Pydantic v1 | T-B001-02 |
| AUD-B-44 | ⚪ | `README.md` | Menciona SQLAlchemy «likely», endpoints sin prefijo y una arquitectura que no existe | T-B001-13 |
| AUD-B-45 | ⚪ | `.dockerignore:14` | Ignora `docker-compose.yml`, pero el fichero es `.yaml` | T-B001-10 |

## 3. Lo que se conserva

- FastAPI + Pydantic v2 y MongoDB como almacenamiento (los datos ya están ahí).
- Las funcionalidades de negocio: registro, login, perfil, historial, informe PDF y asistente.
- La intención de separar `domain` / `infrastructure` / `api` (se completa con `application`).
- Las mediciones legacy de `recognition_results`, que se migran a la nueva colección (B-002).
