import os
import json
import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.errors import HttpError

# Archivo donde se almacenarán los conteos previos
archivo_datos = "conteo_digitos.json"

# Credenciales de Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Obtener lista de imágenes en la carpeta
extensiones_validas = ('.jpg', '.png', '.jpeg')
imagenes = [f for f in os.listdir() if f.endswith(extensiones_validas)]
imagenes.sort()  # Ordenar los archivos para consistencia


# Función para cargar o obtener las credenciales de Google Drive
def obtener_credenciales():
    """Obtiene credenciales para acceder a Google Drive."""
    creds = None
    # El archivo token.pickle almacena las credenciales del usuario
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # Si no tenemos credenciales válidas, se realiza un login de usuario
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Guardar las credenciales para la próxima vez
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds


# Función para subir una imagen a Google Drive
def subir_imagen_a_drive(imagen, folder_id, numero_carpeta):
    """Sube una imagen a la carpeta correspondiente de Google Drive."""
    try:
        creds = obtener_credenciales()
        service = build('drive', 'v3', credentials=creds)

        # Subir la imagen
        file_metadata = {'name': imagen, 'parents': [folder_id]}
        media = MediaFileUpload(imagen, mimetype='image/jpeg')  # Cambiar si usas otro tipo de imágenes
        archivo = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        # Mostrar el número de la carpeta donde se subió
        print(f"Imagen '{imagen}' subida correctamente a la carpeta '{numero_carpeta}'.")
        return True

    except HttpError as error:
        print(f"Ha ocurrido un error al subir la imagen: {error}")
        return False


# Cargar conteos previos si existen
if os.path.exists(archivo_datos):
    with open(archivo_datos, "r") as f:
        conteo_digitos = json.load(f)
else:
    conteo_digitos = {}

# Preguntar por los conteos iniciales solo la primera vez
if not conteo_digitos:
    for i in range(10):  # Para los números del 0 al 9
        num = str(i)
        cantidad = input(f"¿Cuántas imágenes ya tienes etiquetadas como '{num}'? (Si ninguna, escribe 0): ").strip()
        conteo_digitos[num] = int(cantidad) if cantidad.isdigit() else 0

# Identificadores de carpetas en Google Drive (por cada número del 0 al 9)
folders_drive = {
    '0': '1F1kbmtmWD9W77nv2eIdCDaBauQctMyrC',
    '1': '19bugt9aUWBccFgaz_D3u6bHtim2L1OAW',
    '2': '1bHmFUatXMxmENUqmMnlIFfxRG9SW8xtI',
    '3': '1Wd-4Lpbxs5Ha8JptF7knglC5Z4AxoND4',
    '4': '17oZ08u1hc4x6Dwzo4oXLoTg7Q62jpg3g',
    '5': '1V_EtAYjIR7vV7AAT2rfcsgAJPxw34yp-',
    '6': '11fEtR_uoPHabXEHjikP_ZPb9dM6ikaul',
    '7': '14isoiez_TFnt0RxbvZa6V1EULkSZWlXH',
    '8': '1zFL3nxfuN3hY2WHKENYejxyaI987G6nD',
    '9': '14UsC1P5QhiH4QbU4LHuCNFxaoN91lDu8',
}

# Variable para almacenar los números de las imágenes
imagenes_por_numero = {}

# Preguntar por el número de cada imagen solo una vez
for imagen in imagenes:
    print(f"\nImagen: {imagen}")
    digito = input("¿Qué número ves en esta imagen? (0-9): ").strip()

    # Validar entrada
    if not digito.isdigit() or not (0 <= int(digito) <= 9):
        print("Entrada inválida. Debe ser un número entre 0 y 9.")
        continue

    # Almacenar el número para cada imagen
    imagenes_por_numero[imagen] = digito

# Procesar las imágenes
for imagen, digito in imagenes_por_numero.items():
    # Incrementar conteo y renombrar imagen
    conteo_digitos[digito] += 1
    nuevo_nombre = f"{conteo_digitos[digito]}_{digito}{os.path.splitext(imagen)[1]}"
    os.rename(imagen, nuevo_nombre)

    # Subir la imagen a la carpeta correspondiente en Google Drive
    if subir_imagen_a_drive(nuevo_nombre, folders_drive[digito], digito):
        # Si la imagen se sube correctamente, borrar la imagen local
        os.remove(nuevo_nombre)

    # Guardar progreso en el archivo JSON
    with open(archivo_datos, "w") as f:
        json.dump(conteo_digitos, f)

print("\nProceso completado!!")
