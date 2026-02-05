"""Module for evaluation of different detectors."""

from src.configuration import config
from src.data_models import Evaluation, Sample, SubsetName
from src.detection.detector import Detector
from src.training.dataset import DatasetManager, FilesystemDatasetManager


class Evaluator:
    """Evaluator of LLM-written text detectors."""

    def __init__(self, dataset_name: str = config.main_dataset) -> None:
        """
        Configure evaluation dataset used and dataset manager.

        Args:
            dataset_name (str, optional): Name of the dataset to be used.
                Defaults to the value from the configuration.
        """
        self._detectors = {}
        self._dataset_name = dataset_name
        self._dataset_manager: DatasetManager = FilesystemDatasetManager()

    def add_detector(self, detector: Detector) -> None:
        """
        Add a detector to a study.

        Args:
            detector (Detector): An instance of a detector to be evaluated.
        """
        self._detectors[detector.get_name()] = detector

    def _evaluate_detector(
        self, detector: Detector, samples: list[Sample]
    ) -> Evaluation:
        epsilon = 1 ** (-10)
        correctly_predicted = 0
        all_samples = len(samples)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for sample in samples:
            predicted = detector.detect(text=sample.text, language=sample.language)
            actual = sample.to_probability()

            binary_prediction = 1.0 if predicted >= detector.get_threshold() else 0.0

            if binary_prediction == 1.0:
                if actual == 1.0:
                    true_positives += 1
                    correctly_predicted += 1
                else:
                    false_positives += 1
            elif binary_prediction == 0.0:
                if actual == 1.0:
                    false_negatives += 1
                else:
                    correctly_predicted += 1

        accuracy = correctly_predicted / all_samples

        precision_denominator = true_positives + false_positives
        # Prevent division by zero error.
        if precision_denominator == 0:
            precision_denominator = epsilon
        precision = true_positives / precision_denominator

        recall_denominator = true_positives + false_negatives
        # Prevent division by zero error.
        if recall_denominator == 0:
            recall_denominator = epsilon
        recall = true_positives / recall_denominator

        f1_score_denominator = precision + recall
        # Prevent division by zero error.
        if f1_score_denominator == 0:
            f1_score_denominator = epsilon
        f1_score = 2 * (precision * recall) / f1_score_denominator

        return Evaluation(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )

    async def evaluate(self, subset: SubsetName) -> dict[str, Evaluation]:
        """
        Run the evaluation study for all added detectors.

        Args:
            subset (SubsetName): Name of the subset. Either "training", "validation",
                or "testing".

        Returns:
            dict[str, Evaluation]: Mapping of names of detectors to
                their evaluation sheets.
        """
        dataset = await self._dataset_manager.get_raw_dataset(
            dataset_name=self._dataset_name
        )
        samples: list[Sample] = getattr(dataset, subset).samples
        return {
            name: self._evaluate_detector(detector, samples)
            for name, detector in self._detectors.items()
        }
