from fastapi import APIRouter, UploadFile, File, Depends
from app.services.firebase_service import save_prediction

router = APIRouter()