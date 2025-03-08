# Libraries
import cv2
import numpy as np
from skimage import filters, measure
from PIL import Image

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

    def extract_display_area(self,image: np.ndarray) -> np.ndarray:
        """
        Detecta automáticamente el área del display de un tensiómetro en una imagen y la recorta.
        :param image: Imagen de entrada.
        :return: Región de la imagen recortada correspondiente al display. Devuelve None si no se detecta.
        """
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Aplicar desenfoque para suavizar la imagen
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Aplicar un umbral binario o adaptativo para resaltar los contornos
        #_, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        # También puedes probar con umbral adaptativo:
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Asegurarse de que thresh sea del tipo CV_8UC1
        #thresh = np.uint8(thresh)

        # Buscar contornos en la imagen binarizada
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Inicializar el área del display
        display_contour = None
        max_area = 0

        # Encontrar el contorno más grande que probablemente sea el display
        for contour in contours:
            # Calcular el área del contorno
            area = cv2.contourArea(contour)

            # Filtrar contornos pequeños que no sean el display
            if area > 1000:  # Ajusta este valor dependiendo del tamaño esperado del display
                # Aproximar el contorno a un cuadrilátero
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

                # Si es un contorno rectangular (cuatro vértices), tomarlo
                if len(approx) == 4 and area > max_area:
                    display_contour = approx
                    max_area = area

        # Si se encuentra el contorno del display, recortar la región
        if display_contour is not None:
            # Obtener un bounding box alrededor del contorno
            x, y, w, h = cv2.boundingRect(display_contour)

            # Recortar el área del display de la imagen original
            display_area = image[y:y + h, x:x + w]
            return display_area

        # Si no se detecta un área de display, devolver None
        print("No se pudo detectar el área del display.")
        return None

    def detect_digit_positions(self, display_area: np.ndarray) -> list:
        """
        Detects the positions of digits within the display using outlines or segmentation
        :param display_area: image of the display area
        :return: list of coordinates of the detected digits
        """
        # Detect contours
        contours, _ = cv2.findContours(display_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_positions = []
        for contour in contours:
            # Ignore small contours that are not digits
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                digit_positions.append((x, y, w, h))

        # Sort positions
        digit_positions.sort(key=lambda pos: pos[0])

        return digit_positions

    def crop_digit_areas(self, display: np.ndarray) -> list:
        """
        Crops and saves images at the specified coordinates.

        :param display: Input image in numpy array format.
        :return: List with the coordinates of the saved crops.
        """

        # Crop coordinates (x1: left, x2: top, y1: right, y2: bottom)
        coordinates = [
            (140, 160, 440, 620),   # Region 1
            (450, 160, 750, 620),   # Region 2
            (760, 160, 1060, 620),  # Region 3
            (450, 640, 750, 1100),  # Region 4
            (760, 640, 1060, 1100), # Region 5
            (740, 1150, 900, 1400), # Region 6
            (920, 1150, 1080, 1400) # Region 7
        ]

        # Iterate over the coordinates
        cropped_images = []

        # Crop the image region
        for idx, (x1, x2, y1, y2) in enumerate(coordinates):
            # Recortar la región de la imagen
            cropped_img = display[x2:y2, x1:y1]

            # Save the cropped image to a file
            file_name = f"digit_region_{idx}.png"
            cv2.imwrite(file_name, cropped_img)
            print(f"Imagen guardada: {file_name}")
            print(cropped_img.shape)
            # Store the cropped image in the list (optional)
            cropped_images.append(cropped_img)

        # Return the used coordinates (if necessary)
        return coordinates

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


