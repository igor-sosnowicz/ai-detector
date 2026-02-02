"""Module with project-wide data models."""

import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal, Self, override

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.nlp.tokeniser import LLamaTokeniser

Language = Literal["english", "french", "german", "spanish", "polish"]
ChatRole = Literal["assistant", "user", "tool", "system"]
SubsetName = Literal["training", "validation", "testing"]
TensorisedSample = tuple[np.ndarray, np.ndarray]


class TensorConvertible(ABC):
    """An interface for converting an object to a tensor."""

    @abstractmethod
    def to_tensors(self) -> TensorisedSample:
        """
        Convert an object into a tensor.

        Returns:
            TensorisedSample: A tensorised object ready to be stacked to form
                a large tensor for vectorised processing.
        """


class SampleType(Enum):
    """Possible values of a label."""

    FULLY_LLM_WRITTEN = 0
    FULLY_HUMAN_WRITTEN = 1

    def one_hot_encode(self) -> np.ndarray:
        """
        Encode a type of a sample as one-hot vector.

        Returns:
            np.ndarray: One-hot vector.
        """
        vector = [0] * len(SampleType)
        vector[self.value] = 1
        return np.array(vector)


class Sample(BaseModel, TensorConvertible):
    """A sample for training/validating/testing a ML model."""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    label: SampleType
    author: str  # LLM's model or author's name
    prompt_uuid: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @override
    def to_tensors(self) -> TensorisedSample:
        tokeniser = LLamaTokeniser()
        tokens = tokeniser.tokenise(self.text)
        token_ids = tokeniser.encode(tokens)
        return (
            token_ids,
            self.label.one_hot_encode(),
        )


class DataSplitFractions(BaseModel):
    """Fractions representing split of samples across subsets of the dataset."""

    training_fraction: float = 0.75
    validation_fraction: float = 0.1
    testing_fraction: float = 0.15

    @model_validator(mode="after")
    def validate_fractions_sum(self) -> Self:
        """Validate whether fractions sum up to 1.0."""
        total = sum(
            (
                Decimal(str(self.training_fraction)),
                Decimal(str(self.validation_fraction)),
                Decimal(str(self.testing_fraction)),
            )
        )
        if total != 1.0:
            raise ValueError(
                "The sum of fractions representing samples split into subsets "
                f"has to sum up to 1.0 but it sums up to {total}."
            )

        return self


class RawSubset(BaseModel):
    """Subset of the dataset containing samples."""

    name: SubsetName
    samples: list[Sample]

    def get_size(self) -> int:
        """
        Get a number of samples in a subset.

        Returns:
            int:
        """
        return len(self.samples)


class RawDataset(BaseModel):
    """Dataset containing raw samples in training, validation, and testing sets."""

    name: str
    training: RawSubset
    validation: RawSubset
    testing: RawSubset
    split: DataSplitFractions

    model_config = ConfigDict(frozen=True)


class TensorSubset(BaseModel):
    """Subset of the dataset."""

    name: SubsetName
    features: np.ndarray
    labels: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_size(self) -> int:
        """
        Get a number of samples in a subset.

        Returns:
            int:
        """
        return self.features.shape[0]


class TensorDataset(BaseModel):
    """Dataset consisting of training, validation, and testing subsets."""

    name: str
    training: TensorSubset
    validation: TensorSubset
    testing: TensorSubset
    split: DataSplitFractions

    model_config = ConfigDict(frozen=True)


class ChatMessage(BaseModel):
    """Chat message exchanged between a user and an LLM."""

    message: str
    role: ChatRole
    timestamp: datetime = datetime.now(tz=UTC)

    def to_dict(self) -> dict[str, str]:
        """
        Convert a message to a dictionary.

        Returns:
            dict[str, str]: Message as a dictionary.
        """
        return {
            "content": self.message,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
        }
