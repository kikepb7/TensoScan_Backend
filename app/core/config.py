from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Settings(BaseSettings):
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    MODEL_PATH: str = os.getenv("MODEL_PATH", "ia_models/")
    FIREBASE_CREDENTIALS: str = os.getenv("FIREBASE_CREDENTIALS")

settings = Settings()