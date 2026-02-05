"""Module with detection based on the compression."""

from pathlib import Path
from typing import override

import zippy

from src.data_models import Language, Range
from src.detection.detector import Detector


class CompressionBasedDetector(Detector):
    """LLM-generated text detector using characteristic compression ratios of LLMs."""

    def __init__(self) -> None:
        """Initialise path to a file with model."""
        self._persistence_path = (
            Path("./data/models/compression_based_detector_parameters.json")
            .expanduser()
            .resolve()
        )
        self.load()

    @override
    def detect(self, text: str, language: Language) -> float:
        if language != "english":
            raise NotImplementedError("Support for other languages is missing yet.")

        z = zippy.EnsembledZippy()
        score = z.run_on_text_chunked(s=text)
        if score is None:
            raise ValueError("Zippy has returned None.")
        return 0.0 if score[0] == "Human" else 1.0

    @override
    def get_threshold(self) -> float:
        return 1.0

    @override
    def get_tunable_attributes(self) -> dict[str, Range]:
        # It does not make sense to tune its threshold.
        return {}
