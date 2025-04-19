import io, os, cv2, logging
import numpy as np
from PIL import Image
from app.infrastructure.services.image_processor_service import ImageProcessorService
from app.domain.interfaces.predictor_interaface import PredictorInterface
from app.domain.models.display_result_model import DisplayRecognitionResult

logger = logging.getLogger(__name__)

class DisplayRecognizerService:

    def __init__(self, model: PredictorInterface, image_processor: ImageProcessorService):
        self.model = model
        self.image_processor = image_processor

    def recognize_display(self, image_bytes: bytes) -> DisplayRecognitionResult:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            temp_path = os.path.join(os.getcwd(), "display.jpg")
            img.save(temp_path)

            # Cargar con OpenCV
            img_cv = cv2.imread(temp_path)

            display_area = self.image_processor.extract_display_area(img_cv)
            if display_area is None:
                raise ValueError("Could not detect display area")

            display_area = cv2.resize(display_area, (1109, 1431))

            digit_images = self.image_processor.crop_digit_areas(display_area)

            predictions = []
            confidences = []

            # digit_paths = sorted(glob.glob(os.path.join("app", "*.png")))

            for idx, item in enumerate(digit_images):
                # Asegúrate de extraer la imagen solo si viene en tupla (imagen, algo más)
                if isinstance(item, tuple):
                    digit_np = item[0]
                else:
                    digit_np = item

                try:
                    # Validación opcional para asegurarte que es un ndarray
                    if not isinstance(digit_np, np.ndarray):
                        logger.error(f"Item at index {idx} is not a valid image array: {type(digit_np)}")
                        continue

                    digit_pil = Image.fromarray(digit_np)
                    result = self.model.predict(digit_pil)

                    predictions.append(str(result.digit))
                    confidences.append(result.confidence)

                    logger.info(f"Digit {idx} predicted: {result.digit} with confidence: {result.confidence}")

                except Exception as e:
                    logger.error(f"Failed to predict digit at index {idx}: {str(e)}")

            high_pressure = ''.join(predictions[:3])
            low_pressure = ''.join(predictions[3:5])
            pulse = ''.join(predictions[5:])
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return DisplayRecognitionResult(
                high_pressure=high_pressure,
                low_pressure=low_pressure,
                pulse=pulse,
                confidence=avg_conf
            )

        except Exception as e:
            logger.error(f"Error in display recognition service: {str(e)}")
            raise