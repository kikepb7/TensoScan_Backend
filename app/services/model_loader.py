from tensorflow.keras.models import load_model
from app.core.config import settings

# Load IA model once in cache
model = load_model(settings.MODEL_PATH)

def get_ia_model():
    """Return IA model from cache"""
    return model