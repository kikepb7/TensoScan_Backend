from abc import ABC, abstractmethod
from PIL import Image
from app.domain.models.prediction_result_model import PredictionResult


class PredictorInterface(ABC):
    @abstractmethod
    def predict(self, image: Image) -> PredictionResult:
        pass