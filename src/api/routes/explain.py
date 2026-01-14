from fastapi import APIRouter, HTTPException
from src.api.schemas_explain import ExplanationRequest, ExplanationResponse
from src.api.utils import model_loader
from src.api.explainability import explainer_service

router = APIRouter()

@router.post("/explain", response_model=ExplanationResponse)
async def explain_sentiment(request: ExplanationRequest):
    try:
        sentiment, confidence = model_loader.predict(request.text)
        explanation = explainer_service.get_explanation(request.text)
        
        return ExplanationResponse(
            sentiment=sentiment,
            confidence=confidence,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
