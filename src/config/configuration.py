"""
Configuration Manager for modular MLOps project.
- Uses ConfigBox (dot-notation) for config access
- Uses @ensure_annotations for type safety
- Provides ModelTrainerConfig entity with MLflow settings
- Includes custom logger and exception handling
"""
from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Please add 'PyYAML' to requirements.") from e

try:
    from ensure import ensure_annotations  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("ensure is required. Please add 'ensure' to requirements.") from e

try:
    from box import ConfigBox  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("python-box is required. Please add 'python-box' to requirements.") from e

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
    """Custom exception for configuration-related errors."""
    pass


# ---------- Fallback utilities (used until src/utils/common.py is added) ----------

# @ensure_annotations
def _read_yaml(path_to_yaml: str) -> ConfigBox:
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        return ConfigBox(content)
    except FileNotFoundError as fnf:
        logger.error(f"YAML not found at: {path_to_yaml}")
        raise CustomException(fnf)
    except Exception as e:
        logger.exception("Failed to read YAML file")
        raise CustomException(e)

# @ensure_annotations
def _create_directories(path_list: List[str]) -> None:
    try:
        for p in path_list:
            os.makedirs(p, exist_ok=True)
    except Exception as e:
        logger.exception("Failed to create directories")
        raise CustomException(e)

# Try to import project utilities, otherwise use fallbacks defined above
try:
    from src.utils.common import read_yaml as read_yaml  # type: ignore
    from src.utils.common import create_directories as create_directories  # type: ignore
except Exception:
    read_yaml = _read_yaml
    create_directories = _create_directories


# ---------- Entities ----------
# Import required dataclasses from src/entity/config_entity.py
from src.entity.config_entity import (
    ModelTrainerConfig,
    DataIngestionConfig,
    DataTransformationConfig,
)  # type: ignore


# ---------- Configuration Manager ----------

class ConfigurationManager:
    """Reads config.yaml and params.yaml and exposes typed entities.

    Expected YAML structure (example):
    config.yaml:
      artifacts_root: artifacts
      model_trainer:
        model_dir: models
        model_name: logreg_tfidf.pkl
        tracking_uri: mlruns
        experiment_name: imdb-sentiment

    params.yaml:
      model_trainer:
        random_state: 42
        max_iter: 200
        C: 1.0
        solver: lbfgs
        n_jobs: -1
    """

    # @ensure_annotations
    def __init__(self, config_file_path: Optional[str] = None, params_file_path: Optional[str] = None) -> None:
        try:
            # Prefer constants if available
            try:
                from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH  # type: ignore
                default_config = CONFIG_FILE_PATH
                default_params = PARAMS_FILE_PATH
            except Exception:
                default_config = os.path.join("config", "config.yaml")
                default_params = os.path.join("params.yaml")

            self.config_path = config_file_path or default_config
            self.params_path = params_file_path or default_params

            logger.info(f"Reading config from: {self.config_path}")
            self.config: ConfigBox = read_yaml(self.config_path)

            logger.info(f"Reading params from: {self.params_path}")
            self.params: ConfigBox = read_yaml(self.params_path)

            # Normalize top-level keys to ConfigBox
            self.config = ConfigBox(self.config)
            self.params = ConfigBox(self.params)
        except Exception as e:
            logger.exception("Failed to initialize ConfigurationManager")
            raise CustomException(e)

    # @ensure_annotations
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Build and return ModelTrainerConfig entity from YAMLs.
        Creates necessary directories.
        """
        try:
            # Read config with defaults
            c_mt: ConfigBox = ConfigBox(self.config.get("model_trainer", {}))
            p_mt: ConfigBox = ConfigBox(self.params.get("model_trainer", {}))

            artifacts_root: str = self.config.get("artifacts_root", "artifacts")

            model_dir = c_mt.get("model_dir", os.path.join("models"))
            model_name = c_mt.get("model_name", "logreg_tfidf.pkl")
            tracking_uri = c_mt.get("tracking_uri", "mlruns")
            experiment_name = c_mt.get("experiment_name", "imdb-sentiment")
            registered_model_name = c_mt.get("registered_model_name", "sentiment_analysis_model")

            random_state = p_mt.get("random_state", 42)
            max_iter = p_mt.get("max_iter", 200)
            C = p_mt.get("C", 1.0)
            solver = p_mt.get("solver", "lbfgs")
            n_jobs = p_mt.get("n_jobs", -1)
            class_weight = p_mt.get("class_weight", None)

            # Ensure directories exist
            metrics_dir = os.path.join(artifacts_root, "metrics")
            create_directories([artifacts_root, model_dir, metrics_dir])

            logger.info("Constructed ModelTrainerConfig entity")
            return ModelTrainerConfig(
                model_dir=model_dir,
                model_name=model_name,
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                registered_model_name=registered_model_name,
                random_state=int(random_state),
                max_iter=int(max_iter),
                C=float(C),
                solver=str(solver),
                class_weight=class_weight,
                n_jobs=int(n_jobs),
                metrics_dir=metrics_dir,
            )
        except Exception as e:
            logger.exception("Failed to build ModelTrainerConfig")
            raise CustomException(e)

    # @ensure_annotations
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            c_di: ConfigBox = ConfigBox(self.config.get("data_ingestion", {}))
            dataset_path = c_di.get("dataset_path")
            text_column = c_di.get("text_column")
            label_column = c_di.get("label_column")
            label_mapping = c_di.get("label_mapping", None)

            if not dataset_path or not text_column or not label_column:
                raise CustomException("data_ingestion config requires dataset_path, text_column, label_column")

            logger.info("Constructed DataIngestionConfig entity")
            return DataIngestionConfig(
                dataset_path=str(dataset_path),
                text_column=str(text_column),
                label_column=str(label_column),
                label_mapping=label_mapping,
            )
        except Exception as e:
            logger.exception("Failed to build DataIngestionConfig")
            raise CustomException(e)

    # @ensure_annotations
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifacts_root: str = self.config.get("artifacts_root", "artifacts")
            c_dt: ConfigBox = ConfigBox(self.config.get("data_transformation", {}))
            p_dt: ConfigBox = ConfigBox(self.params.get("data_transformation", {}))

            vectorizer_dir = c_dt.get("vectorizer_dir", os.path.join(artifacts_root, "vectorizer"))
            vectorizer_name = c_dt.get("vectorizer_name", "tfidf.pkl")

            test_size = p_dt.get("test_size", 0.2)
            random_state = p_dt.get("random_state", 42)
            max_features = p_dt.get("max_features", None)
            ngram_range = tuple(p_dt.get("ngram_range", [1, 1]))
            lowercase = p_dt.get("lowercase", True)
            stop_words = p_dt.get("stop_words", None)

            # Ensure directories
            create_directories([artifacts_root, vectorizer_dir])

            logger.info("Constructed DataTransformationConfig entity")
            return DataTransformationConfig(
                artifacts_root=artifacts_root,
                test_size=float(test_size),
                random_state=int(random_state),
                max_features=max_features,
                ngram_range=ngram_range,
                lowercase=bool(lowercase),
                stop_words=stop_words,
                vectorizer_dir=vectorizer_dir,
                vectorizer_name=vectorizer_name,
            )
        except Exception as e:
            logger.exception("Failed to build DataTransformationConfig")
            raise CustomException(e)
