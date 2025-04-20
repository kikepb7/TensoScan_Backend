from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # App Settings
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # AI Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "ia_models/")

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # MongoDB
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "tenso_scan_db")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecret")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"

settings = Settings()