"""Module with natural language tokenisers."""

from abc import ABC, abstractmethod
from typing import override


class Tokeniser(ABC):
    """An interface of a natural language tokeniser."""

    @abstractmethod
    def tokenise(self, text: str) -> list[str]:
        """
        Split a text into textual tokens.

        Args:
            text (str): A text to be split.

        Returns:
            list[str]: A list of resulting textual tokens.

        """

    @abstractmethod
    def encode(self, tokens: list[str]) -> list[int]:
        """
        Encode a list of textual tokens into token ids.

        Args:
            tokens (list[str]): A list of textual tokens.

        Returns:
            list[int]: A resulting list of tokens.
        """


class LLamaTokeniser(Tokeniser):
    """Wrapper around Llama fast tokeniser conformat with Tokeniser interface."""

    def __init__(self, max_length: int = 2048) -> None:
        """
        Initialise a Llama tokeniser and configure it.

        Args:
            max_length (int, optional): The maximum length of an output of
                the tokeniser. Defaults to 2048 tokens.
        """
        import logging

        from transformers import LlamaTokenizerFast

        # Suppress tokenizer warnings about long sequences
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        # Initialize tokenizer with truncation
        self._tokeniser = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
        self._max_length = max_length

    @override
    def encode(self, tokens: list[str]) -> list[int]:
        # Truncate if needed
        tokens = tokens[: self._max_length]
        return self._tokeniser.convert_tokens_to_ids(tokens)

    @override
    def tokenise(self, text: str) -> list[str]:
        tokens = self._tokeniser.tokenize(text)
        # Truncate to max length
        return tokens[: self._max_length]
