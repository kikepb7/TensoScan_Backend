import glob
import traceback

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.prediction import NumberRecognizer
from app.utils.image_processing import ImageProcessor
from PIL import Image
import io, logging
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])

image_processor = ImageProcessor()

try:
    logger.info("Initializing NumberRecognizer model...")
    model = NumberRecognizer()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise


@router.post("/recognize")
async def recognize_numbers(image: UploadFile = File(...)):
    try:
        logger.info(f"Processing image: {image.filename}")

        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File uploaded is not an image")

        # Read the image
        contents = await image.read()
        try:
            img = Image.open(io.BytesIO(contents))
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Process image and get prediction
        try:
            result = model.predict(img)
            logger.info(f"Prediction successful: {result}")
            return {
                "prediction": {
                    "digit": result["digit"],
                    "confidence": result["confidence"]
                }
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/display-recognize")
async def display_recognize(image: UploadFile = File(...)):
    try:
        logger.info(f"Processing image: {image.filename}")

        # Validar tipo de archivo
        if not image.content_type.startswith('image/'):
            logger.error("Uploaded file is not an image")
            raise HTTPException(status_code=400, detail="File uploaded is not an image")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, "display.jpg")

        # Leer la imagen
        contents = await image.read()
        try:
            img = Image.open(io.BytesIO(contents))
            img.save(save_path)

            ####
            img2 = cv2.imread(r"C:\TensoScan-Backend\app\api\routes\display.jpg")
            ####

        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Extraemos el display
        display_area = image_processor.extract_display_area(img2)
        display_area = cv2.resize(display_area, (1109, 1431))
        print(display_area.shape)
        if display_area is None:
            raise HTTPException(status_code=400, detail="Could not detect display area")

        # Recortar los dígitos
        digit_images = image_processor.crop_digit_areas(display_area)

        predictions = []
        confidences = []

        # Realizar predicciones
        ####
        digit_images_path = sorted(glob.glob(os.path.join(r"C:\TensoScan-Backend\app", "*.png")))
        ####

        for image in digit_images_path:
            # digit_pil = Image.fromarray(digit)
            digit = Image.open(image)

            result = model.predict(digit)
            predictions.append(str(result["digit"]))
            confidences.append(result["confidence"])

        # Construir las cadenas de presión y pulso
        high_pressure = ''.join(predictions[:3])
        low_pressure = ''.join(predictions[3:5])
        pulse = ''.join(predictions[5:])

        # Promediar la confianza de las predicciones
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        # Devolver los resultados
        return {
            "high_pressure": high_pressure,
            "low_pressure": low_pressure,
            "pulse": pulse,
            "confidence": avg_confidence
        }

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image")


@router.get("/test")
async def test_endpoint():
    return {
        "status": "ok",
        "message": "El servicio de OCR está funcionando correctamente"
    }


@router.get("/test-recognition")
async def test_recognition():
    """
    Endpoint de prueba que simula reconocer números sin necesidad de subir una imagen
    """
    sample_numbers = [4, 2, 8]
    return {
        "status": "ok",
        "detected_numbers": sample_numbers,
    }