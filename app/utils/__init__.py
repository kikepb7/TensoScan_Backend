import os
import cv2
from image_processing import ImageProcessor

def main():
    ruta_imagen = r"C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\029.jpg"
    ruta_salida = os.path.dirname(ruta_imagen)

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

    image_with_digits = processor.detect_and_enclose_digits(display_area)

    # Mostrar la imagen con la rejilla
    cv2.imshow("Imagen con dígitos enmarcados", image_with_digits)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
