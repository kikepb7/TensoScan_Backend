import cv2
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        """
        Initialize image processor
        """
    @staticmethod
    def show_resized_image(window_name: str, image: np.ndarray, max_size: int = 800):
        """
        Muestra una imagen redimensionada si es demasiado grande.
        """
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk and convert it to an array
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Image not found in {image_path}")
            self.show_resized_image("Loaded Image", image)  # Mostrar imagen cargada
            return image
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Aumenta el contraste de la imagen con CLAHE
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
        self.show_resized_image("Enhanced Contrast", enhanced_image)  # Mostrar imagen con contraste mejorado
        return enhanced_image

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica umbralización adaptativa y normalización a la imagen
        """
        image = self.enhance_contrast(image)
        # Probar umbralización fija
        _, adaptive_thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)  # Umbral fijo
        self.show_resized_image("Processed Image", adaptive_thresh)  # Mostrar imagen procesada
        return adaptive_thresh

    def extract_display_area(self, image: np.ndarray, coords: tuple) -> np.ndarray:
        """
        Extrae el área de visualización según las coordenadas proporcionadas
        """
        x1, y1, x2, y2 = coords
        display_area = image[y1:y2, x1:x2]
        self.show_resized_image("Extracted Display Area", display_area)  # Mostrar área extraída
        return display_area

    def detect_digit_positions(self, display_area: np.ndarray) -> list:
        """
        Detecta los contornos de los dígitos en la imagen, filtrando por tamaño y forma
        """
        # Aplica desenfoque gaussiano para reducir ruido
        blurred = cv2.GaussianBlur(display_area, (5, 5), 0)
        # Umbralización binaria inversa para resaltar los números
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

        # Encuentra los contornos de los objetos en la imagen
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_positions = []

        for c in contours:
            # Obtiene el rectángulo delimitador para cada contorno
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)  # Relación de aspecto

            # Filtrar contornos:
            # - Relación de aspecto más cercana a 1 para los números
            # - Área mínima mayor para evitar que se detecten letras pequeñas
            if 0.8 < aspect_ratio < 1.2 and cv2.contourArea(c) > 500:  # Ajuste en relación y área
                digit_positions.append((x, y, w, h))

        # Ordenar los dígitos por la coordenada X (de izquierda a derecha)
        return sorted(digit_positions, key=lambda pos: pos[0])

    def crop_digit(self, display_area: np.ndarray, digit_position: tuple) -> np.ndarray:
        """
        Recorta un dígito en base a sus coordenadas
        """
        x, y, w, h = digit_position
        cropped_digit = display_area[y:y+h, x:x+w]
        self.show_resized_image("Cropped Digit", cropped_digit)  # Mostrar dígito recortado
        return cropped_digit

    def resize_image(self, image: np.ndarray, target_size: tuple = (28, 28)) -> np.ndarray:
        """
        Redimensiona una imagen al tamaño deseado
        """
        resized_image = cv2.resize(image, target_size)
        self.show_resized_image("Resized Image", resized_image)  # Mostrar imagen redimensionada
        return resized_image

    def convert_image_to_pil(self, image: np.ndarray) -> Image:
        """
        Convierte una imagen de NumPy a formato PIL
        """
        pil_image = Image.fromarray(image)
        return pil_image

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica filtro de eliminación de ruido
        """
        denoised_image = cv2.medianBlur(image, 3)
        self.show_resized_image("Denoised Image", denoised_image)  # Mostrar imagen denoised
        return denoised_image
