"""
Model Trainer Component
- Trains Logistic Regression on TF-IDF features
- Logs parameters and metrics to MLflow
- Saves the trained model locally
- Uses @ensure_annotations for type safety and includes custom logger/exception
"""
from __future__ import annotations
import os
import logging
from typing import Any, Tuple, Dict

try:
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required. Please add 'joblib' to requirements.") from e

try:
    import mlflow  # type: ignore
    import mlflow.sklearn  # type: ignore
    from mlflow.models.signature import infer_signature # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("mlflow is required. Please add 'mlflow' to requirements.") from e

try:
    from ensure import ensure_annotations  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("ensure is required. Please add 'ensure' to requirements.") from e

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required. Please ensure 'scikit-learn' is in requirements.") from e

# ---------- Logger ----------

def _get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s"))
        logger.addHandler(ch)
    logger.propagate = False
    return logger

logger = _get_logger(__name__)


# ---------- Custom Exception ----------

class CustomException(Exception):
    """Custom exception for trainer-related errors."""
    pass


# ---------- Entity import (fallback if missing) ----------
try:
    from src.entity.config_entity import ModelTrainerConfig  # type: ignore
except Exception:
    # Minimal fallback mirroring configuration.py
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class ModelTrainerConfig:
        model_dir: str
        model_name: str = "logreg_tfidf.pkl"
        tracking_uri: str = "mlruns"
        experiment_name: str = "imdb-sentiment"
        registered_model_name: str = "sentiment_analysis_model"
        random_state: int = 42
        max_iter: int = 200
        C: float = 1.0
        solver: str = "lbfgs"
        class_weight: Optional[Dict[str, float]] = None
        n_jobs: int = -1
        metrics_dir: str = os.path.join("artifacts", "metrics")


# ---------- Trainer ----------

class ModelTrainer:
    # @ensure_annotations
    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config
        logger.info("Initialized ModelTrainer with config")

    # @ensure_annotations
    def _build_model(self) -> LogisticRegression:
        try:
            model = LogisticRegression(
                C=self.config.C,
                max_iter=self.config.max_iter,
                solver=self.config.solver,
                class_weight=self.config.class_weight,
                n_jobs=self.config.n_jobs,
                random_state=self.config.random_state,
            )
            return model
        except Exception as e:
            logger.exception("Failed to build LogisticRegression model")
            raise CustomException(e)

    # @ensure_annotations
    def train(self,
              X_train: Any,
              y_train: Any,
              X_test: Any,
              y_test: Any) -> Tuple[LogisticRegression, Dict[str, float]]:
        """Train LR model, evaluate on test, log to MLflow, and persist model.
        Returns the trained model and a metrics dictionary.
        """
        try:
            model = self._build_model()

            # Configure MLflow
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)

            with mlflow.start_run(run_name="logreg-tfidf"):
                # Log hyperparameters
                mlflow.log_params({
                    "C": self.config.C,
                    "max_iter": self.config.max_iter,
                    "solver": self.config.solver,
                    "class_weight": self.config.class_weight,
                    "n_jobs": self.config.n_jobs,
                    "random_state": self.config.random_state,
                })

                # Fit model
                logger.info("Training Logistic Regression...")
                model.fit(X_train, y_train)

                # Predictions and metrics
                y_pred = model.predict(X_test)
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average="binary")),
                    "recall": float(recall_score(y_test, y_pred, average="binary")),
                    "f1": float(f1_score(y_test, y_pred, average="binary")),
                }

                # Log metrics
                mlflow.log_metrics(metrics)
                logger.info(f"Evaluation metrics: {metrics}")

                # Infer signature
                signature = infer_signature(X_train, model.predict(X_train))

                # Log and register model to MLflow Model Registry
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.config.registered_model_name,
                    signature=signature,
                    input_example=X_train[:1]
                )

                # Persist locally
                os.makedirs(self.config.model_dir, exist_ok=True)
                model_path = os.path.join(self.config.model_dir, self.config.model_name)
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, artifact_path="model_local")
                logger.info(f"Saved trained model at: {model_path}")

            return model, metrics
        except Exception as e:
            logger.exception("Training failed")
            raise CustomException(e)
