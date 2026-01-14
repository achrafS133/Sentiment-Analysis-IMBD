from __future__ import annotations
import logging
import os
import joblib
from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation

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

class Stage02DataTransformation:
    def __init__(self, config_manager: ConfigurationManager):
        self.cm = config_manager

    def run(self, texts, labels):
        cfg = self.cm.get_data_transformation_config()
        transformer = DataTransformation(cfg)
        X_train, X_test, y_train, y_test = transformer.split_and_vectorize(texts, labels)
        logger.info("Stage 02 complete: data transformation")
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        # Load data from previous stage
        input_path = "artifacts/data_ingestion/data.pkl"
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file {input_path} not found. Run stage 01 first.")
        
        texts, labels = joblib.load(input_path)
        logger.info(f"Loaded {len(texts)} samples from {input_path}")

        config_manager = ConfigurationManager()
        stage = Stage02DataTransformation(config_manager)
        X_train, X_test, y_train, y_test = stage.run(texts, labels)
        
        # Save output for next stage
        os.makedirs("artifacts/data_transformation", exist_ok=True)
        joblib.dump((X_train, X_test, y_train, y_test), "artifacts/data_transformation/data.pkl")
        logger.info("Saved transformed data to artifacts/data_transformation/data.pkl")
        
    except Exception as e:
        logger.exception(e)
        raise e
