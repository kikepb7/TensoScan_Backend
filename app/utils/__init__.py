from app.utils.image_processing import ImageProcessor
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creamos una instancia de la clase
processor = ImageProcessor()

# Llamamos a los métodos pasando sus argumentos
image = processor.load_image("C:/Users/garci/OneDrive/Documentos/_IABD/_proyectos/tensoscan/TensoScan_Images/tensiometro.jpg")

# Mejoramos el preprocesamiento aumentando contraste y aplicando un filtro adaptativo
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

image = enhance_contrast(image)

processed_image = processor.process_image(image)  # Procesar imagen

display_area = processor.extract_display_area(image, (10, 10, 2875, 4100))  # Extraer zona de interés

digit_positions = processor.detect_digit_positions(display_area)  # Detectar posiciones de los dígitos

# Recortar y procesar cada dígito
digits = []
for position in digit_positions:
    digit_image = processor.crop_digit(display_area, position)
    resized_digit = processor.resize_image(digit_image)
    denoised_digit = processor.denoise_image(resized_digit)
    pil_digit = processor.convert_image_to_pil(denoised_digit)
    digits.append(pil_digit)

# Mostrar los dígitos recortados
fig, axes = plt.subplots(1, len(digits), figsize=(10, 3))
for ax, digit, i in zip(axes, digits, range(len(digits))):
    ax.imshow(digit, cmap='gray')
    ax.set_title(f"D {i+1}")
    ax.axis('off')
plt.show()
