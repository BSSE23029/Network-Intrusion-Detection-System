from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: int
    status: str
    confidence: float
    risk_score: float
    risk_level: str