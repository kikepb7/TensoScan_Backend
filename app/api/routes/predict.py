from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.prediction import NumberRecognizer
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])

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