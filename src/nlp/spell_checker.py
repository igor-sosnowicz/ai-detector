"""Package with a dictionary."""

from pathlib import Path
from typing import ClassVar

from spellchecker import SpellChecker

from src.configuration import config


class NLTKSpellChecker:
    """Spell checker with text normalization and custom dictionary."""

    # Translation table for curly quotes to straight quotes.
    QUOTE_MAP: ClassVar[dict[str, str]] = {
        "\u2018": "'",  # left single quotation mark to straight (')
        "\u2019": "'",  # right single quotation mark to straight (')
        "\u201c": '"',  # left double quotation mark to straight (")
        "\u201d": '"',  # right double quotation mark to straight (")
    }

    # Characters to remove
    PUNCTUATION_CHARS: ClassVar[str] = '*,().?!:{}‑—•;|~–"-#»'  # noqa: RUF001

    def __init__(self, dict_file: Path = config.nltk_missing_words) -> None:
        """Initialise with custom dictionary file."""
        self.spell_checker = SpellChecker()
        with dict_file.open("r") as f:
            words = [line.strip() for line in f if line.strip()]
        self.spell_checker.word_frequency.load_words(words)

    def _normalise_text(self, text: str) -> str:
        """Normalize text by mapping special quotes and removing punctuation."""
        # Map curly quotes to straight quotes.
        for old, new in self.QUOTE_MAP.items():
            text = text.replace(old, new)
        # Replace punctuation with spaces (not empty strings) to separate words.
        for char in self.PUNCTUATION_CHARS:
            text = text.replace(char, " ")
        return text

    def count_misspelt(self, text: str) -> int:
        """Find misspelt words in text."""
        normalized = self._normalise_text(text)
        cleaned_words = set()
        for word in normalized.split():
            # Skip words with uppercase letters, digits, or @ symbols.
            if word.lower() != word or any(c.isdigit() for c in word) or "@" in word:
                continue
            # Remove possessive forms and leading/trailing apostrophes.
            cleaned = word.strip("'").removesuffix("'s").removesuffix("'")
            if cleaned:
                cleaned_words.add(cleaned)
        return len(self.spell_checker.unknown(cleaned_words))
