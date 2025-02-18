from fastapi import FastAPI
from app.api.routes import predict

app = FastAPI(title="TensorFlow", version="1.0.0")

# Routes
app.include_router(predict.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings

    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT)