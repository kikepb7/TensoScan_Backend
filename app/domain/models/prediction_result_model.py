from pydantic import BaseModel

class PredictionResult(BaseModel):
    digit: int
    confidence: float