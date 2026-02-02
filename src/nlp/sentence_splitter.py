"""Module for splitting a text into sentences."""

from abc import ABC, abstractmethod
from typing import override

import nltk

from src.data_models import Language


class SentenceSplitter(ABC):
    """Interface for splitting a text into sentences."""

    @abstractmethod
    def split_into_sentences(self, text: str, language: Language) -> list[str]:
        """
        Split a text into sentences.

        Args:
            text (str): Text to be split.
            language (Language): Language of the text.

        Returns:
            list[str]: List of sentences, one items is one sentence.
        """


class NLTKSentenceSplitter(SentenceSplitter):
    """NLTK-based implementation of sentence splitter."""

    def __init__(self) -> None:
        """Download `punkt` data required by `nltk` package for sentence splitting."""
        nltk.download("punkt_tab")

    @override
    def split_into_sentences(self, text: str, language: Language) -> list[str]:
        return nltk.sent_tokenize(text, language=language)
