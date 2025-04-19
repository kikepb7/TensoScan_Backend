from abc import ABC, abstractmethod
import numpy as np

class ImageProcessorInterface(ABC):
    @abstractmethod
    def extract_display_area(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def crop_digit_areas(self, image: np.ndarray) -> list:
        pass