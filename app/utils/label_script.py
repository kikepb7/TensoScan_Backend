import os
import json

# Archivo donde se almacenarán los conteos previos
archivo_datos = "conteo_digitos.json"

# Obtener lista de imágenes en la carpeta
extensiones_validas = ('.jpg', '.png', '.jpeg')
imagenes = [f for f in os.listdir() if f.endswith(extensiones_validas)]
imagenes.sort()  # Ordenar los archivos para consistencia

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

# Procesar las imágenes
for imagen in imagenes:
    print(f"\nImagen: {imagen}")
    digito = input("¿Qué número ves en esta imagen? (0-9): ").strip()

    # Validar entrada
    if not digito.isdigit() or not (0 <= int(digito) <= 9):
        print("Entrada inválida. Debe ser un número entre 0 y 9.")
        continue

    # Incrementar conteo y renombrar imagen
    conteo_digitos[digito] += 1
    nuevo_nombre = f"{conteo_digitos[digito]}_{digito}{os.path.splitext(imagen)[1]}"
    os.rename(imagen, nuevo_nombre)

    print(f"Renombrado: {imagen} -> {nuevo_nombre}")

    # Guardar progreso
    with open(archivo_datos, "w") as f:
        json.dump(conteo_digitos, f)

print("\nProceso completado. ¡Todas las imágenes han sido renombradas!")