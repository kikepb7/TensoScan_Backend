from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client = AsyncIOMotorClient(settings.MONGODB_URL)
db = client[settings.DB_NAME]

users_collection = db["users"]
results_collection = db["recognition_results"]