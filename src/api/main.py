from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import predict, health, explain
from src.api.utils import model_loader
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="API for predicting sentiment of movie reviews",
    version="1.0.0"
)

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(explain.router, tags=["Explainability"])

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting up API service...")
    try:
        model_loader.load_artifacts()
        logger.info("Model and vectorizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        # We don't raise here to allow the app to start, but health check might fail if we added deep checks

@app.get("/")
async def root():
    return {"message": "Welcome to IMDB Sentiment Analysis API. Visit /docs for documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
