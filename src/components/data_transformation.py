from __future__ import annotations
import os
import logging
from typing import List, Tuple, Any

try:
    from ensure import ensure_annotations  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("ensure is required. Please add 'ensure' to requirements.") from e

try:
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required. Please ensure it's in requirements.") from e

try:
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("joblib is required. Please add 'joblib' to requirements.") from e


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


class CustomException(Exception):
    """Custom exception for transformation-related errors."""
    pass


try:
    from src.entity.config_entity import DataTransformationConfig  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing DataTransformationConfig entity.") from e


class DataTransformation:
    # @ensure_annotations
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
        logger.info("Initialized DataTransformation with config")

    # @ensure_annotations
    def split_and_vectorize(self, texts: List[str], labels: List[int]) -> Tuple[Any, Any, List[int], List[int]]:
        try:
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                texts,
                labels,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=labels,
            )

            vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                lowercase=self.config.lowercase,
                stop_words=self.config.stop_words,
            )
            logger.info("Fitting TF-IDF vectorizer...")
            X_train = vectorizer.fit_transform(X_train_texts)
            X_test = vectorizer.transform(X_test_texts)

            # Save vectorizer
            os.makedirs(self.config.vectorizer_dir, exist_ok=True)
            vec_path = os.path.join(self.config.vectorizer_dir, self.config.vectorizer_name)
            joblib.dump(vectorizer, vec_path)
            logger.info(f"Saved TF-IDF vectorizer at: {vec_path}")

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.exception("Failed to split and vectorize")
            raise CustomException(e)
