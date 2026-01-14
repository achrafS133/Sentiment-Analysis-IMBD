
import shap
import json
import logging
import numpy as np
from src.api.utils import model_loader

logger = logging.getLogger("explainability")

class Explainer:
    _instance = None
    _explainer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Explainer, cls).__new__(cls)
        return cls._instance
    
    def get_explanation(self, text: str):
        try:
            model, vectorizer = model_loader.load_artifacts()
            
            # For linear models like LogisticRegression, we can use LinearExplainer
            # But we need the transformed features
            features = vectorizer.transform([text])
            
            # Initialize explainer if not done yet
            # Note: Initializing explainer with full training data is expensive.
            # Here we might need a background dataset or just use the model coefficients for linear models directly?
            # SHAP LinearExplainer works well with sklearn linear models without background data sometimes,
            # but usually needs X_train summary.
            
            # For simplicity in this demo, we'll try to use the model and a zero-background or small background if possible.
            # But since we don't have X_train easily available purely in the API without loading it...
            # We will use a generic workaround or just coefficient inspection for LogReg if SHAP is too heavy.
            
            # BETTER APPROACH:
            # Just return top contributing words based on coefficients for LogReg.
            if hasattr(model, "coef_"):
                feature_names = vectorizer.get_feature_names_out()
                coefs = model.coef_[0]
                
                # Get non-zero indices for this document
                doc_indices = features.nonzero()[1]
                
                contributions = []
                for idx in doc_indices:
                    word = feature_names[idx]
                    score = coefs[idx] * features[0, idx]
                    contributions.append({"word": word, "score": float(score)})
                
                # Sort by absolute score
                contributions.sort(key=lambda x: abs(x["score"]), reverse=True)
                return contributions[:10] # Top 10
            
            return [{"error": "Model not supported for lightweight explanation"}]

        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return [{"error": str(e)}]

explainer_service = Explainer()
