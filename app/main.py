from fastapi import FastAPI
from app.api.routes.ocr_routes import router as ocr_router
from app.api.routes.chatbot_routes import router as chatbot_router
import uvicorn

app = FastAPI(
    title="Number Recognition API",
    description="API para reconocimiento de números en imágenes"
)

app.include_router(ocr_router)
app.include_router(chatbot_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Number Recognition API",
        "endpoints": {
            "ocr_recognize": "/ocr/recognize",
            "ocr_test": "/ocr/test",
            "ocr_test_recognition": "/ocr/test-recognition"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )