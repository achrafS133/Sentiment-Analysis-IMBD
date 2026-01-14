from pydantic import BaseModel, Field
from typing import List, Optional

class SentimentRequest(BaseModel):
    text: str = Field(..., description="Movie review text to analyze", example="This movie was fantastic! best plotting ever.")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative)")
    confidence: float = Field(..., description="Prediction confidence score (0-1)")
    processing_time: float = Field(..., description="Time taken to process in seconds")

class BatchSentimentRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of movie reviews")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
