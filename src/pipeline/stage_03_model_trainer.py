from __future__ import annotations
import logging
import os
import joblib
from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer

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

class Stage03ModelTrainer:
    def __init__(self, config_manager: ConfigurationManager):
        self.cm = config_manager

    def run(self, X_train, X_test, y_train, y_test):
        cfg = self.cm.get_model_trainer_config()
        trainer = ModelTrainer(cfg)
        model, metrics = trainer.train(X_train, y_train, X_test, y_test)
        logger.info("Stage 03 complete: model training")
        return model, metrics

if __name__ == "__main__":
    try:
        # Load data from previous stage
        input_path = "artifacts/data_transformation/data.pkl"
        if not os.path.exists(input_path):
             raise FileNotFoundError(f"Input file {input_path} not found. Run stage 02 first.")
        
        X_train, X_test, y_train, y_test = joblib.load(input_path)
        logger.info(f"Loaded training data from {input_path}")

        config_manager = ConfigurationManager()
        stage = Stage03ModelTrainer(config_manager)
        model, metrics = stage.run(X_train, X_test, y_train, y_test)
        
    except Exception as e:
        logger.exception(e)
        raise e
