from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import MONGODB_URL, DB_NAME

client = AsyncIOMotorClient(MONGODB_URL)
db = client[DB_NAME]

users_collection = db["users"]
results_collection = db["recognition_results"]