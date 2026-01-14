from __future__ import annotations
import logging
import os
import joblib
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion

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

class Stage01DataIngestion:
    def __init__(self, config_manager: ConfigurationManager):
        self.cm = config_manager

    def run(self):
        config = self.cm.get_data_ingestion_config()
        ingestion = DataIngestion(config)
        texts, labels = ingestion.load()
        logger.info("Stage 01 complete: data ingestion")
        return texts, labels

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        stage = Stage01DataIngestion(config_manager)
        texts, labels = stage.run()
        
        # Save output for next stage
        os.makedirs("artifacts/data_ingestion", exist_ok=True)
        joblib.dump((texts, labels), "artifacts/data_ingestion/data.pkl")
        logger.info("Saved ingested data to artifacts/data_ingestion/data.pkl")
        
    except Exception as e:
        logger.exception(e)
        raise e
