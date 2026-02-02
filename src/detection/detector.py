"""Module with an interface for a detector."""

from abc import ABC, abstractmethod

from src.data_models import Language


class Detector(ABC):
    """An interface for a LLM-written text detector."""

    @abstractmethod
    def detect(self, text: str, language: Language) -> float:
        """
        Detect and get probability of a text being LLM-written.

        Args:
            text (str): Text to be evaluated.
            language (Language): Language, in which a text is written.

        Returns:
            float: Probability of the text being LLM-generated.
                1.0 means certainly prepared by an LLM, 0.0 means human-made.
                The value is always in the range [0, 1].
        """
