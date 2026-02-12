"""Abstract base class for all pipeline model services."""

from abc import ABC, abstractmethod
from typing import Any
import logging

logger = logging.getLogger(__name__)


class BaseModelService(ABC):
    """Base class for pipeline model inference services.

    Subclasses must implement:
        - model_name: identifier used in model_registry
        - load(): load model artifacts into memory
        - predict(**kwargs): run inference and return typed schema
    """

    model_name: str = ""
    _loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights/artifacts. Called once by model_registry."""

    @abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """Run inference. Returns a Pydantic schema defined per model."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if not self._loaded:
            logger.info("Loading model: %s", self.model_name)
            self.load()
            self._loaded = True
            logger.info("Model loaded: %s", self.model_name)
