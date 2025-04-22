import io, logging, os
from typing import List
from PIL import Image
from datetime import datetime
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.domain.models.display_result_model import DisplayRecognitionResult, Measurement
from app.infrastructure.services.display_recognizer_service import DisplayRecognizerService
from app.infrastructure.services.keras_number_recognizer import KerasNumberRecognizer
from app.infrastructure.services.image_processor_service import ImageProcessorService
from app.infrastructure.services.get_user_service import get_current_user
from app.infrastructure.database.mongo_database import results_collection


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["OCR"])

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '../ia_models', 'modelo_prueba_keras.keras')
model = KerasNumberRecognizer(MODEL_PATH)
preprocessor = ImageProcessorService()
display_service = DisplayRecognizerService(model, preprocessor)

try:
    logger.info("Initializing NumberRecognizer model...")
    # model = NumberRecognizer()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise


@router.post("/recognize")
async def recognize_numbers(image: UploadFile = File(...)):
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File uploaded is not an image")
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))
        result = model.predict(img)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/display-recognize", response_model=DisplayRecognitionResult)
async def display_recognize(image: UploadFile = File(...), user = Depends(get_current_user)):
    try:
        logger.info(f"Received image: {image.filename}")
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File uploaded is not an image")

        image_bytes = await image.read()
        result = display_service.recognize_display(image_bytes)

        # Save in MongoDB
        await results_collection.insert_one({
            "user_id": ObjectId(user["_id"]),
            "filename": image.filename,
            "result": result.model_dump(),
            "timestamp": datetime.now()
        })

        return result

    except Exception as e:
        logger.error(f"Display recognition failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image")


@router.get("/measurements", response_model=List[Measurement])
async def get_user_measurements(user=Depends(get_current_user)):
    try:
        user_id = user["_id"]
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)

        results_cursor = results_collection.find( {"user_id": user_id} )
        results = await results_cursor.to_list(length=None)

        formatted_results = []
        for measure in results:
            formatted_results.append({
                "filename": measure["filename"],
                "result": {
                    "high_pressure": measure["result"]["high_pressure"],
                    "low_pressure": measure["result"]["low_pressure"],
                    "pulse": measure["result"]["pulse"],
                    "confidence": measure["result"]["confidence"],
                },
                "timestamp": measure["timestamp"]
            })

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving measurements")