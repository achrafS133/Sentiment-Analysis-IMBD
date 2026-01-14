from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ExplanationRequest(BaseModel):
    text: str = Field(..., description="Text to explain")

class ExplanationResponse(BaseModel):
    sentiment: str
    confidence: float
    explanation: List[Dict[str, Any]]
