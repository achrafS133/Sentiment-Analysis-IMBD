from __future__ import annotations
import logging

from src.config.configuration import ConfigurationManager  # type: ignore
from src.pipeline.stage_01_data_ingestion import Stage01DataIngestion  # type: ignore
from src.pipeline.stage_02_data_transformation import Stage02DataTransformation  # type: ignore
from src.pipeline.stage_03_model_trainer import Stage03ModelTrainer  # type: ignore


def _get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s"))
        logger.addHandler(ch)
    logger.propagate = False
    return logger

logger = _get_logger("imdb_sentiment_mlop")


def run_pipeline():
    cm = ConfigurationManager()

    # Stage 01: Ingestion
    s1 = Stage01DataIngestion(cm)
    texts, labels = s1.run()

    # Stage 02: Transformation
    s2 = Stage02DataTransformation(cm)
    X_train, X_test, y_train, y_test = s2.run(texts, labels)

    # Stage 03: Trainer
    s3 = Stage03ModelTrainer(cm)
    model, metrics = s3.run(X_train, X_test, y_train, y_test)

    logger.info(f"Pipeline completed. Metrics: {metrics}")


if __name__ == "__main__":
    run_pipeline()
