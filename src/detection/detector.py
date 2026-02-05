"""Module with an interface for a detector."""

from abc import ABC, abstractmethod

from src.data_models import FilePersistent, Language


class Detector(FilePersistent, ABC):
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

    @abstractmethod
    def get_threshold(self) -> float:
        """
        Get a value above or equal to which a value is a positive samples.

        Returns:
            float: Floating point value threshold for a detector.
        """

    def get_name(self) -> str:
        """
        Get name of the detector.

        Returns:
            str: Name of the detector.
        """
        return type(self).__name__
