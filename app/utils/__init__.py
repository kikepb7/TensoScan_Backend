# from app.utils.image_processing import ImageProcessor
# import matplotlib.pyplot as plt
# import cv2
#
# # Creamos una instancia de la clase
# processor = ImageProcessor()
#
# # Llamamos a los métodos pasando sus argumentos
# image = processor.load_image(r"C:/Users/garci/OneDrive/Documentos/_IABD/_proyectos/tensoscan/TensoScan_Images/tensiometro.jpeg")
#
# # Mejoramos el preprocesamiento aumentando contraste y aplicando un filtro adaptativo
# def enhance_contrast(image):
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     return clahe.apply(image)
#
# image = enhance_contrast(image)
#
# processed_image = processor.process_image(image)  # Procesar imagen
#
# display_area = processor.extract_display_area(processed_image, (10, 10, 2875, 4100))  # Extraer zona de interés
# print(display_area.shape)
#
# digit_positions = processor.detect_digit_positions(display_area)  # Detectar posiciones de los dígitos
# print(digit_positions)
#
# # Recortar y procesar cada dígito
# digits = []
# for position in digit_positions:
#     digit_image = processor.crop_digit(display_area, position)
#     resized_digit = processor.resize_image(digit_image)
#     denoised_digit = processor.denoise_image(resized_digit)
#     pil_digit = processor.convert_image_to_pil(denoised_digit)
#     digits.append(pil_digit)
#
# # Mostrar los dígitos recortados
# fig, axes = plt.subplots(1, len(digits), figsize=(10, 3))
# for ax, digit, i in zip(axes, digits, range(len(digits))):
#     ax.imshow(digit, cmap='gray')
#     ax.set_title(f"D {i+1}")
#     ax.axis('off')
# plt.show()

import os
import cv2
from image_processing_2 import cargar_imagen, recortar_area_principal, generar_regilla, recortar_numeros

def redimensionar_imagen(imagen, ancho=350):
    """Redimensiona la imagen manteniendo la proporción."""
    alto = int((ancho / imagen.shape[1]) * imagen.shape[0])
    return cv2.resize(imagen, (ancho, alto))

def main():
    ruta_imagen = "C:/Users/garci/OneDrive/Documentos/_IABD/_proyectos/tensoscan/TensoScan_Images/tensiometro2.jpg"  # Ajusta la ruta si es necesario
    ruta_salida = os.path.dirname(ruta_imagen)

    imagen = cargar_imagen(ruta_imagen)
    imagen = redimensionar_imagen(imagen)

    # Dibujar marco en la imagen original
    x, y, w, h = 100, 155, 165, 250  # Ajusta según necesidad
    imagen_marco = imagen.copy()
    cv2.rectangle(imagen_marco, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Imagen con Marco", imagen_marco)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Recortar área principal
    recorte = recortar_area_principal(imagen_marco)

    # Generar rejilla sobre el recorte
    imagen_con_rejilla = generar_regilla(recorte)

    # Recortar números individuales
    recortar_numeros(recorte, ruta_salida)

if __name__ == "__main__":
    main()

