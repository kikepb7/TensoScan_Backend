import cv2
import pytesseract
from PIL import Image
import numpy as np
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ImageProcessor:
    def __init__(self, output_folder: str):
        """
        Initialize image processor with an output folder where images will be saved.
        :param output_folder: Directory where the digit images will be saved.
        """
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

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

    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image to enhance OCR recognition.
        :param image: input image
        :return: preprocessed image
        """
        # Apply thresholding (this should make the digits clearer)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilation to make the text more prominent
        dilated = cv2.dilate(thresh, None, iterations=1)

        # Remove noise by applying a median blur
        denoised = cv2.medianBlur(dilated, 3)

        return denoised

    def ocr_digit(self, image: np.ndarray) -> str:
        """
        Use Tesseract to recognize digits in a given image.
        :param image: input image
        :return: recognized text (digit)
        """
        # Convert the image to PIL format for pytesseract compatibility
        pil_image = Image.fromarray(image)

        # Apply Tesseract OCR with configuration to only detect numbers
        detected_text = pytesseract.image_to_string(pil_image, config='--psm 6 -c tessedit_char_whitelist=0123456789')

        # Clean up the text by stripping unwanted characters and spaces
        return detected_text.strip()

    def save_digit_image(self, digit_image: np.ndarray, digit_text: str, index: int):
        """
        Save the preprocessed digit image to the output folder.
        :param digit_image: Image of the digit
        :param digit_text: Text detected for the digit
        :param index: Index to differentiate between digits
        """
        filename = os.path.join(self.output_folder, f"digit_{index}_{digit_text if digit_text else 'unknown'}.png")
        cv2.imwrite(filename, digit_image)

    def detect_and_ocr_digits(self, image: np.ndarray, digit_coordinates: list) -> list:
        """
        Detect and OCR digits in the image by processing each digit using provided coordinates.
        :param image: input image
        :param digit_coordinates: list of coordinates for each digit
        :return: list of recognized digits
        """
        recognized_digits = []

        # Iterate over digit coordinates
        for index, (x, y, w, h) in enumerate(digit_coordinates):
            # Crop the digit from the image
            digit_image = image[y:y + h, x:x + w]

            # Preprocess the digit image for better OCR recognition
            preprocessed_image = self.preprocess_image_for_ocr(digit_image)

            # Use OCR to recognize the digit in the cropped image
            digit_text = self.ocr_digit(preprocessed_image)

            # Save the image of the digit
            self.save_digit_image(preprocessed_image, digit_text, index)

            # Append the recognized digit
            recognized_digits.append(digit_text)

        return recognized_digits

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

    def resize_image(self, image: np.ndarray, target_size: tuple = (28, 28)) -> np.ndarray:
        """
        Resize an image to a specific size
        :param image: input image
        :param target_size: target size, default (28, 28)
        :return: resized image
        """
        return cv2.resize(image, target_size)


def main():
    # Paths
    ruta_imagen = r"C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\tensiometro.jpg"
    ruta_salida = r"C:\Users\garci\OneDrive\Documentos\_IABD\_proyectos\tensoscan\TensoScan_Images\output"

    processor = ImageProcessor(output_folder=ruta_salida)

    # Load the image
    image = processor.load_image(ruta_imagen)

    # Resize image for better visualization
    image = processor.resize_image(image, (800, 800))

    # Extract the display area
    display_area = processor.extract_display_area(image)
    if display_area is None:
        print("No se pudo detectar el área del display.")
        return

    # Coordinates of the digits
    digit_coordinates = [
        (80, 50, 135, 155),  # First digit
        (215, 50, 135, 155),  # Second digit
        (355, 50, 135, 155),  # Third digit
        (215, 210, 135, 155),  # Fourth digit
        (355, 210, 135, 155),  # Fifth digit
        (360, 382, 75, 90),  # Sixth digit
        (435, 382, 75, 90)  # Seventh digit
    ]

    # Detect and OCR the digits
    recognized_digits = processor.detect_and_ocr_digits(display_area, digit_coordinates)

    # Show the recognized digits
    print("Dígitos reconocidos:", recognized_digits)


if __name__ == "__main__":
    main()
