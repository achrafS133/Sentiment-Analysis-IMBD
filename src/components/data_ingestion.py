from __future__ import annotations
import logging
from typing import List, Tuple, Dict, Any

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required. Please ensure it's in requirements.") from e

try:
    from ensure import ensure_annotations  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("ensure is required. Please add 'ensure' to requirements.") from e


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
    """Custom exception for ingestion-related errors."""
    pass


try:
    from src.entity.config_entity import DataIngestionConfig  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing DataIngestionConfig entity.") from e


class DataIngestion:
    # @ensure_annotations
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
        logger.info("Initialized DataIngestion with config")

    # @ensure_annotations - Removed due to compatibility issues with List/Tuple return types
    def load(self) -> Tuple[List[str], List[int]]:
        """Load dataset and return texts and numeric labels."""
        try:
            df = pd.read_csv(self.config.dataset_path)
            if self.config.text_column not in df.columns or self.config.label_column not in df.columns:
                raise CustomException(
                    f"Required columns not found. Have: {list(df.columns)}; "
                    f"need text='{self.config.text_column}', label='{self.config.label_column}'"
                )

            texts: List[str] = df[self.config.text_column].astype(str).tolist()
            raw_labels: List[Any] = df[self.config.label_column].tolist()

            # Map labels if mapping provided; else try to coerce to int
            if self.config.label_mapping:
                mapping: Dict[str, int] = self.config.label_mapping
                labels: List[int] = []
                for v in raw_labels:
                    key_lower = str(v).lower()
                    mapped = mapping.get(key_lower)
                    if mapped is None:
                        mapped = mapping.get(str(v))
                    if mapped is None:
                        try:
                            mapped = int(v)
                        except Exception:
                            raise CustomException(f"Unknown label value: {v}. Provide label_mapping in config.yaml.")
                    labels.append(int(mapped))
            else:
                labels = []
                for v in raw_labels:
                    if isinstance(v, str):
                        lv = v.lower()
                        if lv in ("positive", "pos", "1"):
                            labels.append(1)
                        elif lv in ("negative", "neg", "0"):
                            labels.append(0)
                        else:
                            raise CustomException(f"Unknown string label: {v}. Provide label_mapping in config.yaml.")
                    else:
                        labels.append(int(v))

            print(f"DEBUG: str is {str}, type: {type(str)}")
            logger.info(f"Loaded {len(texts)} samples")
            return texts, labels
        except Exception as e:
            logger.exception("Failed to load dataset")
            # raise CustomException(e)
            raise e
