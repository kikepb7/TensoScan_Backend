import cv2
import numpy as np
from PIL import Image
from skimage import filters, measure

class ImageProcessor:
    def __init__(self):
        """
        Initialize image processor
        """

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk and convert it to an array
        :param image_path: image path
        :return: image loaded and converted in a NumPy array format
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Image not found in {image_path}")
            return image
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")


    def process_image(self, image: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Process image, convert to grayscale, apply thresholding and normalization
        :param image: input image
        :param threshold: threshold value for binarization
        :return: preprocessed image
        """

        # Thresholding
        _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # Normalize
        normalized_image = thresholded_image.astype(np.float32) / 255.0

        return normalized_image

    def extract_display_area(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta automáticamente el área del display de un tensiómetro en una imagen y la recorta.
        :param image: Imagen de entrada.
        :return: Región de la imagen recortada correspondiente al display. Devuelve None si no se detecta.
        """
        # Verifica si la imagen ya está en escala de grises
        if len(image.shape) == 3:  # Tiene más de un canal (es a color)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # Ya es escala de grises

        gray = cv2.equalizeHist(gray)

        # Aplicar desenfoque para suavizar la imagen
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Aplicar umbral adaptativo para resaltar los contornos
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Buscar contornos en la imagen binarizada
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Inicializar el área del display
        display_contour = None
        max_area = 0

        # Encontrar el contorno más grande que probablemente sea el display
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Ajusta este valor según el tamaño esperado del display
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4 and area > max_area:
                    display_contour = approx
                    max_area = area

        # Si se encuentra el contorno del display, recortar la región
        if display_contour is not None:
            x, y, w, h = cv2.boundingRect(display_contour)
            display_area = image[y:y + h, x:x + w]
            return display_area

        print("No se pudo detectar el área del display.")
        return None

    def enclose_digits_with_coordinates(self, image: np.ndarray, digit_coordinates: list) -> np.ndarray:
        """
        Dibuja rectángulos alrededor de los dígitos utilizando las coordenadas proporcionadas.
        :param image: Imagen de entrada
        :param digit_coordinates: Lista de coordenadas de los dígitos en formato (x, y, w, h)
        :return: Imagen con los dígitos enmarcados
        """
        # Crear una copia de la imagen para no modificar la original
        image_with_rectangles = image.copy()

        # Iterar sobre las coordenadas de los dígitos
        for (x, y, w, h) in digit_coordinates:
            # Dibujar un rectángulo alrededor de cada dígito detectado
            cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image_with_rectangles

    def detect_and_enclose_digits(self, image: np.ndarray) -> np.ndarray:
        """
        Detecta los dígitos en la imagen y los enmarca usando coordenadas predefinidas.
        :param image: Imagen de entrada
        :return: Imagen con los dígitos enmarcados
        """
        # Ejemplo de coordenadas para los 7 dígitos a enmarcar
        digit_coordinates = [
            (80, 50, 135, 155),  # Coordenadas del primer dígito (x, y, w, h)
            (215, 50, 135, 155),  # Segundo dígito
            (355, 50, 135, 155),  # Tercer dígito
            (215, 210, 135, 155),  # Segundo dígito
            (355, 210, 135, 155),  # Quinto dígito
            (360, 382, 75, 90),  # Sexto dígito
            (435, 382, 75, 90)  # Séptimo dígito
        ]

        # Enmarcar los dígitos usando las coordenadas proporcionadas
        image_with_digits = self.enclose_digits_with_coordinates(image, digit_coordinates)

        return image_with_digits

    def resize_image(self, image: np.ndarray, target_size: tuple = (28, 28)) -> np.ndarray:
        """
        Resize an image to a specific size
        :param image: input image
        :param target_size: target size, default (28, 28)
        :return: resized image
        """
        image = cv2.resize(image, (300,300))
        cv2.imshow("image", image)
        cv2.waitKey(0)
        return cv2.resize(image, target_size)


    def convert_image_to_pil(self, image: np.ndarray) -> Image:
        """
        Convert numpy image into PIL format
        :param image: image in NumPy format
        :return: image in PIL image
        """
        return Image.fromarray(image)


    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image
        :param image: input image
        :return:  cleaned
        """
        return cv2.medianBlur(image, 3)


