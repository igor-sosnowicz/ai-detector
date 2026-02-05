"""Module with detector based on finding spelling mistakes."""

from pathlib import Path
from typing import override

from nltk.tokenize import RegexpTokenizer

from src.data_models import Language, Range
from src.detection.detector import Detector
from src.nlp.spell_checker import NLTKSpellChecker


class SpellingDetector(Detector):
    """Detector based on misspelling rates of words among people."""

    def __init__(self) -> None:
        """Initialise tokeniser for splitting text into words and spellchecker."""
        self._spell_checker = NLTKSpellChecker()
        self._tokeniser = RegexpTokenizer(r"\w+")
        # Derived empirically from observational studies.
        self._human_error_rate = 0.015
        self._threshold = 0.51
        self._persistence_path = (
            Path("./data/models/spelling_detector_parameters.json")
            .expanduser()
            .resolve()
        )
        self.load()

    @override
    def get_tunable_attributes(self) -> dict[str, Range]:
        return {
            "_human_error_rate": Range(min=0.001, max=0.05, type=float),
            "_threshold": Range(min=0.2, max=0.75, type=float),
        }

    def _count_words(self, text: str) -> int:
        return len(self._tokeniser.tokenize(text))

    def _score_error_rate(
        self, error_rate: float, human_error_rate: float = 0.015
    ) -> float:
        # It should satisfy:
        # - f(x=0) = 0.5
        # - f(x=human_error_rate) = 0.75
        # - f(x=2*human_error_rate) = 1
        # - For x > 2*human_error_rate: f(x) = 1
        return min(1.0, 0.5 + 0.25 * error_rate / human_error_rate)

    @override
    def detect(self, text: str, language: Language) -> float:
        if language != "english":
            raise ValueError(
                f"{SpellingDetector.__name__} currently support only English language."
            )
        misseplt_words = self._spell_checker.count_misspelt(text)
        words = self._count_words(text)

        error_rate = misseplt_words / words
        return self._score_error_rate(error_rate, self._human_error_rate)

    @override
    def get_threshold(self) -> float:
        return self._threshold
