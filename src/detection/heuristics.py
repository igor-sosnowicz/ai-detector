"""Module with heuristic checks for LLM generated content."""

from pathlib import Path
from re import search
from typing import override

import emoji

from src.configuration import config
from src.data_models import Language, Range
from src.detection.detector import Detector
from src.nlp.sentence_splitter import NLTKSentenceSplitter


class Heuristics(Detector):
    """Heuristic-based approach to detecting LLM-written text."""

    # Based on Wikipedia guidelines: https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing

    def __init__(self, language: Language) -> None:
        """Initialise utilities required to run heuristic algorithms."""
        self._phrases = config.suspicious_phrases_file.read_text().splitlines()
        self._sentence_splitter = NLTKSentenceSplitter()
        self._language: Language = language
        self._persistence_path = (
            Path("./data/models/heuristic_detector_parameters.json")
            .expanduser()
            .resolve()
        )

        self._threshold = 0.6
        self._calculate_em_dashes_frequency_weight = 1.0
        self._lacks_is_or_are_weight = 2.0
        self._has_not_only_but_weight = 1.0
        self._has_its_not_just_its_about_weight = 1.0
        self._contains_emoji_weight = 0.9
        self._contains_curly_symbols_weight = 1.5
        self._calculate_suspicious_phrases_frequency_weight = 15.0

        self.load()

    @override
    def get_tunable_attributes(self) -> dict[str, Range]:
        return {
            "_threshold": Range(min=0.0, max=1.0, type=float),
            "_calculate_em_dashes_frequency_weight": Range(
                min=0.2, max=2.0, type=float
            ),
            "_lacks_is_or_are_weight": Range(min=0.5, max=2.0, type=float),
            "_has_not_only_but_weight": Range(min=0.5, max=2.0, type=float),
            "_has_its_not_just_its_about_weight": Range(min=0.5, max=2.0, type=float),
            "_contains_emoji_weight": Range(min=0.5, max=2.0, type=float),
            "_contains_curly_symbols_weight": Range(min=0.5, max=2.0, type=float),
            "_calculate_suspicious_phrases_frequency_weight": Range(
                min=2.0, max=30.0, type=float
            ),
        }

    @override
    def get_threshold(self) -> float:
        return self._threshold

    @override
    def detect(self, text: str, language: Language) -> float:
        if language != "english":
            raise ValueError("Currently, only `english` is supported as a language.")
        # 1.0: AI, 0.0: Human.
        sentences = self._sentence_splitter.split_into_sentences(
            text, language=self._language
        )
        self._sentence_count = len(sentences)

        check_weights = {
            self._calculate_em_dashes_frequency: (
                self._calculate_em_dashes_frequency_weight
            ),
            self._lacks_is_or_are: self._lacks_is_or_are_weight,
            self._has_not_only_but: self._has_not_only_but_weight,
            self._has_its_not_just_its_about: self._has_its_not_just_its_about_weight,
            self._contains_emoji: self._contains_emoji_weight,
            self._contains_curly_symbols: self._contains_curly_symbols_weight,
            self._calculate_suspicious_phrases_frequency: (
                self._calculate_suspicious_phrases_frequency_weight
            ),
        }
        score = 0.0
        total_weight = 0.0
        for check, weight in check_weights.items():
            score += float(check(text)) * weight
            total_weight += weight

        # Some heuristics return score >1.0. Normalise to [0, 1].
        return min(1.0, score / total_weight)

    def _calculate_em_dashes_frequency(self, text: str) -> float:
        return text.count("—") / self._sentence_count

    def _lacks_is_or_are(self, text: str) -> float:
        lowercase_text = text.lower()
        return float("is" in lowercase_text or "are" in lowercase_text)

    def _has_not_only_but(self, text: str) -> float:
        return float(search(r"not only.+but", text.lower()) is not None)

    def _has_its_not_just_its_about(self, text: str) -> float:
        return float(search(r"not just about.+it's", text.lower()) is not None)

    def _contains_emoji(self, text: str) -> float:
        return float(bool(emoji.emoji_count(text)))

    def _contains_curly_symbols(self, text: str) -> float:
        curly_characters = "“”‘’’"  # noqa: RUF001, these marks are being checked for.
        return float(any((character in text) for character in curly_characters))

    def _calculate_suspicious_phrases_frequency(self, text: str) -> float:
        lowercased_text = text.lower()
        suspicious_phrases = sum(
            int(suspicious_phrase in lowercased_text)
            for suspicious_phrase in self._phrases
        )
        return suspicious_phrases / self._sentence_count
