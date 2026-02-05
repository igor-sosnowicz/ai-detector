"""Module for tuning configurable detectors."""

import asyncio
import copy
from numbers import Number

import nest_asyncio
import optuna

from src.detection.detector import Detector
from src.training.evaluation import Evaluator

nest_asyncio.apply()


def optimise(detector: Detector, epochs: int = 100) -> dict[str, Number]:
    """
    Optimise a detector hyperparameters by fine-tuning it.

    Args:
        detector (Detector): The detector to be fine-tuned.
        epochs (int): A number of tuning epochs. Defaults to 100.

    Returns:
        dict[str, Number]: Mapping of attributes of the detector to its optimal values.
    """

    def objective(trial: optuna.Trial) -> float:
        local_detector = copy.deepcopy(detector)
        evaluator = Evaluator()
        evaluator.add_detector(local_detector)

        for (
            attribute_name,
            attribute_range,
        ) in local_detector.get_tunable_attributes().items():
            if attribute_range.type is float:
                setattr(
                    local_detector,
                    attribute_name,
                    trial.suggest_float(
                        attribute_name,
                        low=attribute_range.min,  # pyright: ignore[reportArgumentType]
                        high=attribute_range.max,  # pyright: ignore[reportArgumentType]
                    ),
                )
            elif attribute_range.type is int:
                setattr(
                    local_detector,
                    attribute_name,
                    trial.suggest_int(
                        attribute_name,
                        low=attribute_range.min,  # pyright: ignore[reportArgumentType]
                        high=attribute_range.max,  # pyright: ignore[reportArgumentType]
                    ),
                )

        evaluation = asyncio.run(evaluator.evaluate(subset="training"))
        return evaluation[local_detector.get_name()].f1_score

    # Skip non-tunable detectors.
    if not detector.get_tunable_attributes():
        return {}

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=epochs)

    return study.best_params  # E.g. {'x': 2.002108042}
