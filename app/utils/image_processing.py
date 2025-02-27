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


    def extract_display_area(self, image: np.ndarray, coords: tuple) -> np.ndarray:
        """
        Extract display zone using coordinates (ROI)
        :param image: input image
        :param coords: coordinates that define the display area
        :return: image cropped to the display
        """
        x1, y1, x2, y2 = coords
        display_area = image[y1:y2, x1:x2]
        return display_area


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


