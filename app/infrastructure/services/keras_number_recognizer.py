import cv2
import numpy as np
from PIL import Image
import os, logging
from tensorflow.keras.models import load_model
from app.domain.interfaces.predictor_interaface import PredictorInterface
from app.domain.models.prediction_result_model import PredictionResult

logger = logging.getLogger(__name__)

class KerasNumberRecognizer(PredictorInterface):
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = load_model(model_path)
        self.input_shape = self.model.input_shape
        logger.info(f"Model loaded. Input shape: {self.input_shape}")

    def preprocess_image(self, image: Image) -> np.ndarray:
        img = image.convert('L')
        img = np.array(img)
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (28, 28))
        img = cv2.bitwise_not(img)
        _, img = cv2.threshold(img, 78, 120, cv2.THRESH_BINARY_INV)
        img = img.reshape(1, 28, 28, 1)
        return img

    def predict(self, image: Image) -> PredictionResult:
        processed = self.preprocess_image(image)
        prediction = self.model.predict(processed)
        digit = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][digit])
        return PredictionResult(digit=digit, confidence=confidence)
