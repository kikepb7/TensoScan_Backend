from datetime import datetime
from pydantic import BaseModel

class DisplayRecognitionResult(BaseModel):
    high_pressure: str
    low_pressure: str
    pulse: str
    confidence: float

class Measurement(BaseModel):
    measurement_id: str
    user_id: str
    filename: str
    result: DisplayRecognitionResult
    timestamp: datetime