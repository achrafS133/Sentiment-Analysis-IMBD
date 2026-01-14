from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class DataIngestionConfig:
    dataset_path: str
    text_column: str
    label_column: str
    label_mapping: Optional[Dict[str, int]] = None


@dataclass
class DataTransformationConfig:
    artifacts_root: str
    test_size: float = 0.2
    random_state: int = 42
    # TF-IDF parameters
    max_features: Optional[int] = None
    ngram_range: Tuple[int, int] = (1, 1)
    lowercase: bool = True
    stop_words: Optional[str] = None
    # Persistence
    vectorizer_dir: str = os.path.join("artifacts", "vectorizer")
    vectorizer_name: str = "tfidf.pkl"


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
