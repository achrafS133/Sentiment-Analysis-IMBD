from __future__ import annotations
import os
import json
import logging
from typing import Any, List, Dict

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
    """Custom exception for utility-related errors."""
    pass


# @ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file) or {}
        logger.info(f"YAML loaded: {path_to_yaml}")
        return ConfigBox(content)
    except FileNotFoundError as fnf:
        logger.error(f"YAML not found at: {path_to_yaml}")
        raise CustomException(fnf)
    except Exception as e:
        logger.exception("Failed to read YAML file")
        raise CustomException(e)


# @ensure_annotations
def create_directories(path_list: List[str]) -> None:
    try:
        for path in path_list:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory ensured: {path}")
    except Exception as e:
        logger.exception("Failed to create directories")
        raise CustomException(e)


# @ensure_annotations
def save_json(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON saved: {path}")
    except Exception as e:
        logger.exception("Failed to save JSON")
        raise CustomException(e)
