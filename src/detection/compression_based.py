"""Module with detection based on the compression."""

from typing import override

import zippy

from src.data_models import Language
from src.detection.detector import Detector


class CompressionBasedDetector(Detector):
    """LLM-generated text detector using characteristic compression ratios of LLMs."""

    @override
    def detect(self, text: str, language: Language) -> float:
        if language != "english":
            raise NotImplementedError("Support for other languages is missing yet.")

        z = zippy.EnsembledZippy()
        score = z.run_on_text_chunked(s=text)
        if score is None:
            raise ValueError("Zippy has returned None.")
        return 0.0 if score[0] == "Human" else 1.0
