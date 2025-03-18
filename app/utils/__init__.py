import cv2
from image_processing import ImageProcessor

def main():
    ruta_imagen = r"C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\048.jpg"
    ruta_salida = r"C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\images_output"
    coords = (1250, 1700, 1000, 1250)  # (x, y, ancho, alto)

    processor = ImageProcessor()

    cropped_image_path = processor.extract_display_area_coords(ruta_imagen, coords, ruta_salida)
    print("Imagen recortada guardada en:", cropped_image_path)

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
        (73, 45, 69, 71),  # First digit
        (139, 45, 69, 71),  # Second digit
        (205, 45, 69, 71),  # Third digit
        (139, 119, 69, 71),  # Second digit
        (205, 119, 69, 71),  # Fifth digit
        (206, 198, 33, 47),  # Second digit
        (249, 198, 33, 47)  # Seventh digit
    ]

    processor.crop_and_save_digits(display_area, digit_coordinates, ruta_salida)

if __name__ == "__main__":
    main()
