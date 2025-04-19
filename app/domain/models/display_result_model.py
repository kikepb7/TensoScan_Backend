from pydantic import BaseModel

class DisplayRecognitionResult(BaseModel):
    high_pressure: str
    low_pressure: str
    pulse: str
    confidence: float