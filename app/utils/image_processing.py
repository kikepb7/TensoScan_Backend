import cv2
import numpy as np
from PIL import Image
import os

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

    def extract_display_area_coords(self, ruta_imagen: str, coords: tuple, ruta_salida: str) -> str:
        """
        Crops a specific area of an image and saves it in an output folder.
        :param ruta_imagen: Path to the input image
        :param coords: Tuple with coordinates (x, y, w, h) of the area to crop
        :param ruta_salida: Folder where the cropped image will be saved
        :return: Path to the saved cropped image
        """
        # Cargar la imagen original
        image = cv2.imread(ruta_imagen)
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")

        # Obtener el tamaño original de la imagen
        original_height, original_width = image.shape[:2]

        # Redimensionar la imagen a 800x800 para mejor visualización
        image_resized = cv2.resize(image, (800, 800))

        # Ajustar las coordenadas del área de recorte a la nueva escala (800x800)
        scale_x = 800 / original_width
        scale_y = 800 / original_height

        x, y, w, h = coords

        # Redimensionar las coordenadas del área de recorte a la nueva escala
        x_resized = int(x * scale_x)
        y_resized = int(y * scale_y)
        w_resized = int(w * scale_x)
        h_resized = int(h * scale_y)

        # Dibujar un rectángulo alrededor del área seleccionada en la imagen redimensionada
        image_with_rectangle = image_resized.copy()
        cv2.rectangle(image_with_rectangle, (x_resized, y_resized), (x_resized + w_resized, y_resized + h_resized), (0, 255, 0), 2)  # Rectángulo verde

        # Mostrar la imagen con el marco
        cv2.imshow("Imagen con marco", image_with_rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Recortar la imagen original usando las coordenadas originales
        cropped_image = image[y:y + h, x:x + w]

        # Crear el directorio de salida si no existe
        os.makedirs(ruta_salida, exist_ok=True)
        count = len([name for name in os.listdir(ruta_salida) if name.startswith("recorte_") and name.endswith(".jpg")])
        output_path = os.path.join(ruta_salida, f"display_{count + 1}.jpg")

        # Guardar la imagen recortada
        cv2.imwrite(output_path, cropped_image)
        return output_path

    def extract_display_area(self, image: np.ndarray) -> np.ndarray:
        """
        Detects the display area of a tensiometer in an image and crops it.
        :param image: input image
        :return: Cropped region corresponding to the display area. Returns None if no display area is detected.
        """
        # Check if the image is already in grayscale
        if len(image.shape) == 3:  # Has more than one channel (color image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image  # It's already in grayscale

        gray = cv2.equalizeHist(gray)

        # Apply blur to smooth the image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding to highlight contours
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize the display area
        display_contour = None
        max_area = 0

        # Find the largest contour which is likely the display
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust this value based on the expected display size
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4 and area > max_area:
                    display_contour = approx
                    max_area = area

        # If the display contour is found, crop the region
        if display_contour is not None:
            x, y, w, h = cv2.boundingRect(display_contour)
            display_area = image[y:y + h, x:x + w]
            return display_area

        print("No se pudo detectar el área del display.")
        return None

    def enclose_digits_with_coordinates(self, image: np.ndarray, digit_coordinates: list) -> np.ndarray:
        """
        Draw rectangles around digits using provided coordinates.
        :param image: input image
        :param digit_coordinates: list of digit coordinates in (x, y, w, h) format
        :return: image with rectangles around digits
        """
        # Create a copy of the image to avoid modifying the original
        image_with_rectangles = image.copy()

        # Iterate over digit coordinates and draw rectangles
        for (x, y, w, h) in digit_coordinates:
            # Draw a rectangle around each detected digit
            cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image_with_rectangles

    def detect_and_enclose_digits(self, image: np.ndarray) -> np.ndarray:
        """
        Detects digits in the image and draws rectangles around them using predefined coordinates.
        :param image: input image
        :return: image with digits enclosed in rectangles
        """
        # Example coordinates for 7 digits to be enclosed
        digit_coordinates = [
            (73, 45, 69, 71),  # First digit
            (139, 45, 69, 71),  # Second digit
            (205, 45, 69, 71),  # Third digit
            (139, 119, 69, 71),  # Second digit
            (205, 119, 69, 71),  # Fifth digit
            (206, 198, 33, 47),  # Second digit
            (249, 198, 33, 47)  # Seventh digit
        ]

        # Enclose the digits using the provided coordinates
        image_with_digits = self.enclose_digits_with_coordinates(image, digit_coordinates)

        return image_with_digits

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
        Convert NumPy image to PIL format
        :param image: image in NumPy format
        :return: image in PIL format
        """
        return Image.fromarray(image)

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image
        :param image: input image
        :return: cleaned image
        """
        return cv2.medianBlur(image, 3)

    def crop_and_save_digits(self, image: np.ndarray, digit_coordinates: list, output_path: str):
        """
        Crop each digit from the image and save them as individual files,
        ensuring that the numbering continues from the highest existing number.
        :param image: input image
        :param digit_coordinates: list of coordinates for each digit
        :param output_path: directory where the cropped digit images will be saved
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Find the highest existing digit number to continue the numbering
        existing_files = os.listdir(output_path)
        existing_numbers = []
        for file in existing_files:
            if file.startswith("digit_") and file.endswith(".png"):
                try:
                    # Extract the number from the filename (digit_1.png => 1)
                    number = int(file.split('_')[1].split('.')[0])
                    existing_numbers.append(number)
                except ValueError:
                    continue

        # Determine the next available digit number (continue the numbering)
        next_digit_number = max(existing_numbers, default=0) + 1

        # Iterate over the coordinates and crop each digit
        for i, (x, y, w, h) in enumerate(digit_coordinates):
            # Crop the digit from the image
            digit_image = image[y:y + h, x:x + w]

            # Generate a unique file name for each digit
            output_filename = os.path.join(output_path, f"digit_{next_digit_number}.jpg")

            # Save the cropped digit image
            cv2.imwrite(output_filename, digit_image)
            print(f"Digit {next_digit_number} saved as {output_filename}")

            # Increment the digit number for the next iteration
            next_digit_number += 1