import os
import cv2
from image_processing import ImageProcessor

def main():
    ruta_imagen = r"C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\tensiometro.jpg"
    ruta_salida = r"C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\output"

    processor = ImageProcessor()

    # Cargar imagen
    imagen = processor.load_image(ruta_imagen)

    # Redimensionar imagen para mejor visualización
    imagen = processor.resize_image(imagen, (800, 800))

    # Detectar y recortar área del display
    display_area = processor.extract_display_area(imagen)
    if display_area is None:
        print("No se pudo detectar el área del display.")
        return

    # Detectar y enmarcar los dígitos en la imagen
    image_with_digits = processor.detect_and_enclose_digits(display_area)

    # Mostrar la imagen con los dígitos enmarcados
    cv2.imshow("Imagen con dígitos enmarcados", image_with_digits)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Recortar y guardar cada dígito
    digit_coordinates = [
        (80, 50, 135, 155),  # Primer dígito
        (215, 50, 135, 155),  # Segundo dígito
        (355, 50, 135, 155),  # Tercer dígito
        (215, 210, 135, 155),  # Cuarto dígito
        (355, 210, 135, 155),  # Quinto dígito
        (360, 382, 75, 90),  # Sexto dígito
        (435, 382, 75, 90)  # Séptimo dígito
    ]

    processor.crop_and_save_digits(display_area, digit_coordinates, ruta_salida)

if __name__ == "__main__":
    main()
