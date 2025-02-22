import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumberRecognizer:
    def __init__(self):
        try:
            # Check if model file exists
            rute_model = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'ia_models', "modelo_digitos.tflite")
            self.model_path = os.path.join(rute_model)
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            # Load TFLite model directly
            logger.info(f"Loading model from {self.model_path}")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Get model input shape
            self.input_shape = self.input_details[0]['shape']
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def preprocess_image(self, image: Image):
        try:
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')

            # Resize to match input shape
            target_size = (self.input_shape[1], self.input_shape[2])
            image = image.resize(target_size)

            # Convert to numpy array and normalize
            img_array = np.array(image)
            img_array = img_array.astype('float32') / 255.0

            # Expand dimensions to match the model's expected input shape
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            logger.info(f"Image preprocessed successfully. Shape: {img_array.shape}")
            return img_array

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, image: Image):
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            logger.info("Image preprocessing completed")

            # Set the input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
            logger.info("Input tensor set")

            # Run inference
            self.interpreter.invoke()
            logger.info("Model inference completed")

            # Get prediction results
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_digit = int(np.argmax(predictions[0]))  # Convert numpy.int64 to Python int

            logger.info(f"Prediction successful. Predicted digit: {predicted_digit}")
            return {"digit": predicted_digit, "confidence": float(predictions[0][predicted_digit])}

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise