"""Module with project-wide data models."""

import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self, override

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.nlp.tokeniser import LLamaTokeniser

Language = Literal["english", "french", "german", "spanish", "polish"]
ChatRole = Literal["assistant", "user", "tool", "system"]
SubsetName = Literal["training", "validation", "testing"]
TensorisedSample = tuple[np.ndarray, np.ndarray]
Number = float | int


class Range(BaseModel):
    """Inclusive continuous range of numeric values."""

    min: Number
    max: Number
    type: type[Number]


class FilePersistent(ABC):
    """An interface for object that support persistence in a disk file."""

    _persistence_path: Path

    @abstractmethod
    def get_tunable_attributes(self) -> dict[str, Range]:
        """
        Get a mapping of tunable attributes with their ranges.

        Returns:
            dict[str, Range]: Mapping of tunable attributes with their ranges of valid
                values.
        """

    def set_tunable_attributes(self, attributes: dict[str, Number]) -> None:
        """
        Set tunable attributes to specific values.

        Parameter `attributes` matches perfectly output of the optimise() function
        and should to set optimal parameter values for the detector. Setting individual
        parameters can and should be done via usual dot syntax.

        Args:
            attributes (dict[str, Number]): Mapping of attribute names to their values
                that should be set.

        Raises:
            ValueError: Raised if there is an attempt to set a parameter that does not
                exist in the detector.
        """
        for attribute, value in attributes.items():
            if not hasattr(self, attribute):
                raise ValueError(
                    f"There is no such parameter `{attribute}` for {self.__qualname__}."
                )
            setattr(self, attribute, value)

    def save(self) -> None:
        """Save current values of model parameters to a file."""
        key_value_pairs = {
            attribute: getattr(self, attribute)
            for attribute in self.get_tunable_attributes()
        }
        self._persistence_path.write_text(json.dumps(key_value_pairs))

    def load(self) -> None:
        """Load model parameters from a file to memory."""
        if not self._persistence_path.exists():
            raise FileNotFoundError(
                f"There is no such model file {self._persistence_path} for "
                f"{type(self).__qualname__} to load it."
            )

        parameters: dict[str, Any] = json.loads(self._persistence_path.read_text())
        self.set_tunable_attributes(parameters)

    @classmethod
    def delete(cls) -> None:
        """Delete the file preserving the instance."""
        cls._persistence_path.unlink(missing_ok=True)


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
    language: Language = "english"

    @override
    def to_tensors(self) -> TensorisedSample:
        tokeniser = LLamaTokeniser()
        tokens = tokeniser.tokenise(self.text)
        token_ids = tokeniser.encode(tokens)
        return (
            token_ids,
            self.label.one_hot_encode(),
        )

    def to_probability(self) -> float:
        """
        Convert a label to float point value.

        Returns:
            float: Floating point value of the label.
        """
        if self.label == SampleType.FULLY_HUMAN_WRITTEN:
            return 0.0
        return 1.0


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


class Evaluation(BaseModel):
    """Evaluation sheet for a detector."""

    accuracy: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)

    def __str__(self) -> str:
        """
        Convert the evaluation into a textual form.

        Returns:
            str: Pretty textual form of an evaluation.
        """
        return (
            f"  Accuracy:  {self.accuracy:.4f}\n"
            f"  Recall:  {self.recall:.4f}\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  F1 Score:  {self.f1_score:.4f}"
        )
