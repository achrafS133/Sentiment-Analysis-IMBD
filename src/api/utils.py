
import os
import json
import joblib
import glob
import logging
from typing import Tuple, Any

# Configure logger
logger = logging.getLogger("api_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class ModelLoader:
    _instance = None
    _model = None
    _vectorizer = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_artifacts(self) -> Tuple[Any, Any]:
        """Loads the best model and vectorizer from the latest registry."""
        if self._model is not None and self._vectorizer is not None:
            return self._model, self._vectorizer

        try:
            # 1. Find latest registry file
            registry_files = glob.glob(os.path.join(MODELS_DIR, "*json"))
            if not registry_files:
                raise RuntimeError(f"No registry files found in {MODELS_DIR}")
            
            # Sort by modification time to get the latest
            latest_registry = max(registry_files, key=os.path.getctime)
            logger.info(f"Loading registry: {latest_registry}")

            with open(latest_registry, 'r', encoding='utf-8') as f:
                registry = json.load(f)

            # 2. Find best model (highest F1) in the registry
            models_dict = registry.get("modeles", {})
            if not models_dict:
                raise RuntimeError("No models found in registry")

            # models_dict is { "Model Name": { "f1": 0.88, "chemin": "..." }, ... }
            best_model_name = max(models_dict, key=lambda k: models_dict[k].get("f1", 0))
            best_model_info = models_dict[best_model_name]
            
            model_rel_path = best_model_info["chemin"]
            vectorizer_rel_path = registry.get("chemin_vectorizer")

            # Fix paths (handle windows backslashes if needed, though python usually handles / fine, 
            # but incoming data has backslashes)
            model_path = os.path.join(PROJECT_ROOT, model_rel_path)
            vectorizer_path = os.path.join(PROJECT_ROOT, vectorizer_rel_path)

            logger.info(f"Loading best model: {best_model_name} from {model_path}")
            logger.info(f"Loading vectorizer from {vectorizer_path}")

            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(vectorizer_path):
                 raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

            self._model = joblib.load(model_path)
            self._vectorizer = joblib.load(vectorizer_path)
            self._model_name = best_model_name
            
            return self._model, self._vectorizer

        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    def predict(self, text: str) -> Tuple[str, float]:
        """Predicts sentiment and confidence."""
        model, vectorizer = self.load_artifacts()
        
        # Transform
        features = vectorizer.transform([text])
        
        # Predict
        prediction = model.predict(features)[0]
        # handle different types of labels (int 0/1 or str "positive"/"negative")
        
        # Get probabilities if supported
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            confidence = float(max(probs))
        else:
            # Fallback for models like SVM/SGD if probability=False
            confidence = 1.0 # Placeholder if not available
        
        sentiment_label = "positive" if str(prediction).lower() in ["1", "pos", "positive"] else "negative"
        
        return sentiment_label, confidence

model_loader = ModelLoader()
