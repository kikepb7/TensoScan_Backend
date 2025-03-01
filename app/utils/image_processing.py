# Libraries
import cv2
import numpy as np
from PIL import Image
from skimage import filters, measure


class ImageProcessor:
    def __init__(self):
        """
        Initialize image processor
        """

    @staticmethod
    def show_resized_image(window_name: str, image: np.ndarray, max_size: int = 800):
        """
        Muestra una imagen redimensionada si es demasiado grande.
        :param window_name: Nombre de la ventana.
        :param image: Imagen en formato NumPy.
        :param max_size: Tamaño máximo en píxeles del lado más grande.
        """
        h, w = image.shape[:2]

        # Si la imagen es más grande que max_size, la redimensionamos
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
        :param image_path: image path
        :return: image loaded and converted in a NumPy array format
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise FileNotFoundError(f"Image not found in {image_path}")

            # ImageProcessor.show_resized_image("Loaded Image", image)
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

        # Normaliza, pero convierte a uint8 antes de devolver la imagen
        normalized_image = (thresholded_image.astype(np.float32) / 255.0 * 255).astype(np.uint8)
        # Redimensionar la imagen si es muy grande
        return normalized_image


    def extract_display_area(self, image: np.ndarray, coords: tuple) -> np.ndarray:
        """
        Extract display zone using coordinates (ROI)
        :param image: input image
        :param coords: coordinates that define the display area
        :return: image cropped to the display
        """
        x1, y1, x2, y2 = coords
        display_area = image[y1:y2, x1:x2]
        self.show_resized_image("Display Area", display_area)
        return display_area

    def detect_digit_positions(self, display_area: np.ndarray) -> list:
        """
        Detects the positions of digits within the display using outlines or segmentation
        :param display_area: image of the display area
        :return: list of coordinates of the detected digits
        """
        # Verificar si la imagen tiene 3 canales (BGR)
        if len(display_area.shape) == 3:
            # Convertir a escala de grises si tiene 3 canales
            gray = cv2.cvtColor(display_area, cv2.COLOR_BGR2GRAY)
        else:
            # Si ya está en escala de grises, no hacer nada
            gray = display_area

        # El resto del código sigue igual
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        x, y, w, h = cv2.boundingRect(contours[0])
        screen_roi = display_area[y:y + h, x:x + w]

        digit_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                digit_positions.append((x, y, w, h))

        digit_positions.sort(key=lambda pos: pos[0])

        self.show_resized_image("Contour", screen_roi)

        return digit_positions

    def crop_digit(self, display_area: np.ndarray, digit_position: tuple) -> np.ndarray:
        """
        Crops a specific digit from the display area using the coordinates of the digit
        :param display_area: image of the display area
        :param digit_position: coordinates of the digit to crop
        :return: cropped image of the digit
        """
        x, y, w, h = digit_position
        digit_image = display_area[y:y+h, x:x+w]
        return digit_image


    def resize_image(self, image: np.ndarray, target_size: tuple = (28, 28)) -> np.ndarray:
        """
        Resize an image to a specific size
        :param image: input image
        :param target_size: target size, default (28, 28)
        :return: resized image
        """
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