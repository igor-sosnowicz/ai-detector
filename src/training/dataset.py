"""Module with dataset management tools."""

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final, override

import numpy as np
from loguru import logger

from src.configuration import config
from src.data_models import (
    DataSplitFractions,
    RawDataset,
    RawSubset,
    Sample,
    TensorDataset,
    TensorSubset,
)
from src.training.sample_generator import MixedSampleGenerator


class DatasetManager(ABC):
    """Tool for managing datasets and performing raw-to-tensor dataset conversion."""

    @abstractmethod
    async def get_tensor_dataset(
        self,
        dataset_name: str,
    ) -> TensorDataset:
        """
        Get a tensorised version of the dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            TensorDataset: Tensorised version of the dataset.
        """

    @abstractmethod
    async def get_raw_dataset(
        self,
        dataset_name: str,
    ) -> RawDataset:
        """
        Get a dataset with raw (untensorised) samples.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            RawDataset: Untensorised version of the dataset.
        """

    @abstractmethod
    async def create_dataset_if_missing(
        self,
        dataset_name: str,
        samples: int,
        subset_split: DataSplitFractions | None = None,
    ) -> None:
        """
        Create a dataset and fill it with samples if it is missing.

        Args:
            dataset_name (str): Name of the dataset.
            samples (int): A number of samples to be generated.
            subset_split (DataSplitFractions | None, optional): How to split samples
                among subsets. If None, the default split is used. Defaults to None.
        """

    @abstractmethod
    async def add_samples(
        self,
        dataset_name: str,
        samples: list[Sample],
    ) -> None:
        """
        Add samples to an existing dataset.

        Args:
            dataset_name (str): Name of the dataset to which samples should be added.
            samples (list[Sample]): A list of samples to be added.

        Raises:
            FileNotFoundError: Raised if the dataset does not exist or lacks index file
                containing the split ratios for the subsets of a dataset.
        """


class FilesystemDatasetManager(DatasetManager):
    """Manager for creating, extending, and loading local filesystem datasets."""

    testing_directory: Final = "testing"
    training_validation_directory: Final = "training_validation"
    split_index_file: Final = "index.json"

    def __init__(self, dataset_directory: Path = config.datasets_directory) -> None:
        """
        Set the path to a root directory with datasets in the local filesystem.

        Args:
            dataset_directory (Path, optional): Path to the root directory with dataset.
            Defaults to the value from the configuration.
        """
        self._loaded_datasets: dict[str, TensorDataset] = {}
        self._loaded_raw_datasets: dict[str, RawDataset] = {}
        dataset_directory.mkdir(parents=True, exist_ok=True)
        self._datasets_dir = dataset_directory
        self._split_fractions: dict[str, DataSplitFractions] = {}

    def _ensure_dataset_exist(self, dataset_name: str) -> None:
        path = self._datasets_dir / dataset_name
        if not path.exists():
            raise FileNotFoundError(
                f"The directory for the dataset `{dataset_name}` in "
                f"{path.expanduser().resolve()} does not exist. Create it first by "
                f"calling {self.create_dataset_if_missing.__name__}(...) method."
            )

    def _load_split_proportions(self, dataset_name: str) -> DataSplitFractions:
        # It caches already read indices.
        # Reads content of the index containing the split proportions.
        if dataset_name in self._split_fractions:
            return self._split_fractions[dataset_name]

        path = (
            (self._datasets_dir / dataset_name / self.split_index_file)
            .expanduser()
            .resolve()
        )
        if not path.exists():
            raise FileNotFoundError(
                f"The index file for {dataset_name} is missing. "
                f"It should be located in {path}"
            )

        json_content = path.read_text()
        dict_args = json.loads(json_content)
        split = DataSplitFractions(**dict_args)

        self._split_fractions[dataset_name] = split
        return split

    @override
    async def add_samples(
        self,
        dataset_name: str,
        samples: list[Sample],
    ) -> None:
        self._ensure_dataset_exist(dataset_name)
        path = self._datasets_dir / dataset_name

        subset_split = self._load_split_proportions(dataset_name)
        if subset_split is None:
            subset_split = DataSplitFractions()

        random.shuffle(samples)

        # Testing set cannot be used during any training.
        # Information which samples belong to the testing set should be preserved
        # in a persistent way. Hence, different directories.
        training_validation_samples, testing_samples = self._split_into_subsets(
            samples, subset_split
        )

        # Save samples in the filesystem in the JSON format.
        for sample in training_validation_samples:
            filename = f"{sample.uuid}.json"
            file = path / self.training_validation_directory / filename
            file.write_text(data=sample.model_dump_json())

        for sample in testing_samples:
            filename = f"{sample.uuid}.json"
            file = path / self.testing_directory / filename
            file.write_text(data=sample.model_dump_json())

        tensor_dataset, raw_dataset = await self._load_dataset(dataset_name)
        self._loaded_datasets[dataset_name] = tensor_dataset
        self._loaded_raw_datasets[dataset_name] = raw_dataset

    def _split_into_subsets(
        self, samples: list[Sample], subset_split: DataSplitFractions
    ) -> tuple[list[Sample], list[Sample]]:
        split_index = int((1 - subset_split.testing_fraction) * len(samples))
        return (
            samples[:split_index],
            samples[split_index:],
        )

    @override
    async def get_tensor_dataset(self, dataset_name: str) -> TensorDataset:
        tensor_dataset = self._loaded_datasets.get(dataset_name, None)
        if tensor_dataset is not None:
            return tensor_dataset

        tensor_dataset, raw_dataset = await self._load_dataset(dataset_name)
        self._loaded_datasets[dataset_name] = tensor_dataset
        self._loaded_raw_datasets[dataset_name] = raw_dataset
        return tensor_dataset

    @override
    async def get_raw_dataset(self, dataset_name: str) -> RawDataset:
        raw_dataset = self._loaded_raw_datasets.get(dataset_name, None)
        if raw_dataset is not None:
            return raw_dataset

        raw_dataset = await self._load_raw_dataset_only(dataset_name)
        self._loaded_raw_datasets[dataset_name] = raw_dataset
        return raw_dataset

    async def _load_raw_dataset_only(self, dataset_name: str) -> RawDataset:
        """Load only raw dataset without tensorizing."""
        self._ensure_dataset_exist(dataset_name)

        # Read samples from the filesystem.
        train_val_dir = (
            self._datasets_dir / dataset_name / self.training_validation_directory
        )
        train_val_samples = []
        for file in train_val_dir.glob("*.json"):
            dict_model = json.loads(file.read_text())
            train_val_samples.append(Sample(**dict_model))

        split = self._load_split_proportions(dataset_name)
        split_index = int(
            (
                split.training_fraction
                / (split.training_fraction + split.validation_fraction)
            )
            * len(train_val_samples)
        )
        training_samples = train_val_samples[:split_index]
        validation_samples = train_val_samples[split_index:]

        test_dir = self._datasets_dir / dataset_name / self.testing_directory
        testing_samples = []
        for file in test_dir.glob("*.json"):
            dict_model = json.loads(file.read_text())
            testing_samples.append(Sample(**dict_model))

        # Construct `RawDataset` instance with raw samples.
        return RawDataset(
            name=dataset_name,
            training=RawSubset(
                name="training",
                samples=training_samples,
            ),
            validation=RawSubset(
                name="validation",
                samples=validation_samples,
            ),
            testing=RawSubset(
                name="testing",
                samples=testing_samples,
            ),
            split=split,
        )

    async def _load_dataset(
        self, dataset_name: str
    ) -> tuple[TensorDataset, RawDataset]:
        """Load both tensor and raw datasets (with tensorization)."""
        # First load raw dataset
        raw_dataset = await self._load_raw_dataset_only(dataset_name)

        # Then tensorize it
        training_features, training_labels = self._tensorise_raw_samples(
            raw_dataset.training.samples
        )
        validation_features, validation_labels = self._tensorise_raw_samples(
            raw_dataset.validation.samples
        )
        testing_features, testing_labels = self._tensorise_raw_samples(
            raw_dataset.testing.samples
        )

        # Construct `TensorDataset` instance.
        tensor_dataset = TensorDataset(
            name=dataset_name,
            training=TensorSubset(
                name="training", features=training_features, labels=training_labels
            ),
            validation=TensorSubset(
                name="validation",
                features=validation_features,
                labels=validation_labels,
            ),
            testing=TensorSubset(
                name="testing", features=testing_features, labels=testing_labels
            ),
            split=raw_dataset.split,
        )

        return tensor_dataset, raw_dataset

    @override
    async def create_dataset_if_missing(
        self,
        dataset_name: str,
        samples: int,
        subset_split: DataSplitFractions | None = None,
    ) -> None:
        path = self._datasets_dir / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        index_file = path / self.split_index_file

        samples_directory1 = path / self.training_validation_directory
        samples_directory2 = path / self.testing_directory

        # Check if dataset exists and has enough samples
        tv_files = (
            list(samples_directory1.glob("*.json"))
            if samples_directory1.exists()
            else []
        )
        test_files = (
            list(samples_directory2.glob("*.json"))
            if samples_directory2.exists()
            else []
        )
        total_samples = len(tv_files) + len(test_files)

        if index_file.exists() and total_samples >= samples:
            logger.info(
                f"The dataset `{dataset_name}` already exists with {total_samples} "
                "samples. Skipping its generation."
            )
            return
        if total_samples > 0 and total_samples < samples:
            # Generate only the missing samples
            missing_samples = samples - total_samples
            logger.info(
                f"The dataset `{dataset_name}` has {total_samples} samples but needs "
                f"{samples}. Generating {missing_samples} additional samples..."
            )

            sample_generator = MixedSampleGenerator()
            new_raw_samples = await sample_generator.generate_samples(missing_samples)

            # Use add_samples to append to existing dataset
            await self.add_samples(dataset_name=dataset_name, samples=new_raw_samples)
            return

        # Dataset doesn't exist, create from scratch
        if subset_split is None:
            subset_split = DataSplitFractions()
        index_file.write_text(subset_split.model_dump_json())

        sample_generator = MixedSampleGenerator()
        raw_samples = await sample_generator.generate_samples(samples)
        val_train_samples, test_samples = self._split_into_subsets(
            raw_samples, subset_split
        )

        self._save_dataset(
            dataset_name=dataset_name,
            validation_and_training_samples=val_train_samples,
            testing_samples=test_samples,
        )

        tensor_dataset, raw_dataset = await self._load_dataset(dataset_name)
        self._loaded_datasets[dataset_name] = tensor_dataset
        self._loaded_raw_datasets[dataset_name] = raw_dataset

    def _save_dataset(
        self,
        dataset_name: str,
        validation_and_training_samples: list[Sample],
        testing_samples: list[Sample],
    ) -> None:
        path = self._datasets_dir / dataset_name
        train_val_dir = path / self.training_validation_directory
        test_dir = path / self.testing_directory

        train_val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for sample in validation_and_training_samples:
            sample_path = train_val_dir / f"{sample.uuid}.json"
            sample_path.write_text(sample.model_dump_json())

        for sample in testing_samples:
            sample_path = test_dir / f"{sample.uuid}.json"
            sample_path.write_text(sample.model_dump_json())

    def _tensorise_raw_samples(
        self, raw_samples: list[Sample]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not raw_samples:
            # Handle empty samples case
            return (np.array([]), np.array([]))

        all_features: list[list[int]] = []
        all_labels = []
        for raw_sample in raw_samples:
            features, label = raw_sample.to_tensors()
            all_features.append(features)
            all_labels.append(label)

        # Pad/truncate to a fixed length to allow stacking
        max_len = max(len(f) for f in all_features)
        max_len = min(max_len, 2048)

        tensorised_features = np.zeros((len(all_features), max_len), dtype=np.int32)
        for i, feats in enumerate(all_features):
            trimmed = feats[:max_len]
            tensorised_features[i, : len(trimmed)] = np.asarray(trimmed, dtype=np.int32)

        tensorised_labels = np.stack(all_labels, axis=0)

        return (tensorised_features, tensorised_labels)
