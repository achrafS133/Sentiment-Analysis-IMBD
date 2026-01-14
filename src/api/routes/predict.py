from fastapi import APIRouter, HTTPException
from src.api.schemas import SentimentRequest, SentimentResponse, BatchSentimentRequest, BatchSentimentResponse
from src.api.utils import model_loader
import time

router = APIRouter()

@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        start_time = time.time()
        sentiment, confidence = model_loader.predict(request.text)
        process_time = time.time() - start_time
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            processing_time=process_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-predict", response_model=BatchSentimentResponse)
async def batch_predict_sentiment(request: BatchSentimentRequest):
    try:
        results = []
        for text in request.reviews:
            start_time = time.time()
            sentiment, confidence = model_loader.predict(text)
            process_time = time.time() - start_time
            results.append(SentimentResponse(
                sentiment=sentiment,
                confidence=confidence,
                processing_time=process_time
            ))
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
