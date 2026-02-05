"""Module with a text binary classifier for stylometric features."""

from pathlib import Path
from pickle import dump, load
from typing import override

import numpy as np
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.configuration import config
from src.data_models import Language, Range, RawDataset
from src.detection.detector import Detector
from src.training.dataset import DatasetManager, FilesystemDatasetManager


class StylometricClassifier(Detector):
    """A logistic classifier based on stylometric feature of a text."""

    def __init__(self, model_path: Path = config.stylometric_classifier_path) -> None:
        """
        Initialise a ML model and text vectoriser.

        Args:
            model_path (Path, optional): _description_. Path to the ML model file.
                Defaults to the value from configuration.
        """
        self._model_path = model_path.expanduser().resolve()
        self._vectorizer: TfidfVectorizer | None = None
        self._binary_classifier: GradientBoostingClassifier | None = None
        self._threshold = 0.6
        self._vectoriser_ngram_range_min = 2
        self._vectoriser_ngram_range_max = 5

        # Ignore terms appearing in more than 95% of documents.
        self._vectoriser_max_df = 0.95
        # Ignore terms appearing in two or less documents.
        self._vectoriser_min_df = 2.0

    async def prepare_model(self) -> None:
        """Either load a model if exists or get a dataset and train the ML model."""
        if self._model_file_exists():
            logger.info(
                f"Loading {StylometricClassifier.__name__} from {self._model_path}"
            )
            self._binary_classifier = self._load_model()
        else:
            logger.info(
                f"{StylometricClassifier.__name__} model file does not exist. "
                "Training the model..."
            )
            accuracy = await self._train()
            self._save_model()
            logger.info(f"Successfully trained the model. Accuracy: {accuracy}")

    async def test(self) -> dict[str, float]:
        """
        Evaluate the model on the testing set.

        Returns:
            dict[str, float]: Dictionary containing various metrics:
                - accuracy: Overall accuracy
                - precision: Precision for detecting AI-generated text
                - recall: Recall for detecting AI-generated text
                - f1: F1-score for detecting AI-generated text
                - confusion_matrix: [[TN, FP], [FN, TP]]
        """
        if self._vectorizer is None or self._binary_classifier is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        dataset = await self._load_dataset()
        if dataset.testing.get_size() == 0:
            raise ValueError("Dataset has no testing samples. Cannot test the model.")

        test_texts = [sample.text for sample in dataset.testing.samples]
        test_labels = [sample.label.value for sample in dataset.testing.samples]

        x_test = self._vectorizer.transform(test_texts)
        y_pred = self._binary_classifier.predict(x_test)

        # Calculate metrics (class 0 is AI-generated, class 1 is human)
        metrics = {
            "accuracy": float(accuracy_score(test_labels, y_pred)),
            "precision": float(precision_score(test_labels, y_pred, pos_label=0)),
            "recall": float(recall_score(test_labels, y_pred, pos_label=0)),
            "f1": float(f1_score(test_labels, y_pred, pos_label=0)),
        }

        # Add confusion matrix
        cm = confusion_matrix(test_labels, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Log detailed classification report
        logger.info(
            "\nClassification Report:\n"
            + classification_report(
                test_labels,
                y_pred,
                target_names=["AI (0)", "Human (1)"],
                digits=4,
            )
        )

        # Log confusion matrix
        logger.info(
            f"\nConfusion Matrix:\n"
            f"                 Predicted Human  Predicted AI\n"
            f"Actual Human     {cm[0][0]:<15}  {cm[0][1]}\n"
            f"Actual AI        {cm[1][0]:<15}  {cm[1][1]}\n"
        )

        return metrics

    def __call__(self, text: str) -> float:
        """
        Evaluate whether the text is LLM-written or human-wrriten.

        Args:
            text (str): Text to be evaluated.

        Raises:
            RuntimeError: Raised if the model had not been prepared before
                running this method.

        Returns:
            float: Whether a text was LLM-written. Value in the range [0, 1].
                1.0 means certainly LLM-generated, 0.0 - certainly human-crafted.
        """
        # 0.0 - AI, 1.0 - human (but we return inverted: 1.0 = AI, 0.0 = human).
        if self._vectorizer is None or self._binary_classifier is None:
            raise RuntimeError(
                f"Model not prepared. Call `{self.prepare_model.__name__}` first."
            )
        x = self._vectorizer.transform([text])
        probabilities = self._binary_classifier.predict_proba(X=x)
        return float(probabilities[0, 0])  # Get probability of class 0 (AI)

    @override
    def detect(self, text: str, language: Language) -> float:
        if language != "english":
            raise ValueError(
                "Only English is supported right now for "
                f"{StylometricClassifier.__name__}."
            )
        return self.__call__(text=text)

    @override
    def get_threshold(self) -> float:
        return self._threshold

    @override
    def get_tunable_attributes(self) -> dict[str, Range]:
        return {
            "_threshold": Range(max=1.0, min=0.0, type=float),
            "_vectoriser_ngram_range_min": Range(min=1, max=4, type=int),
            "_vectoriser_ngram_range_max": Range(min=4, max=7, type=int),
            # Both max_df and min_df can take both float (as percentage of documents)
            # and int (as a concrete number of documents).
            "_vectoriser_max_df": Range(min=0.90, max=1.0, type=float),
            "_vectoriser_min_df": Range(min=1, max=5, type=int),
        }

    def explain_prediction(
        self, text: str, top_n: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Explain which features influenced the prediction for a given text.

        Args:
            text (str): Text to explain.
            top_n (int): Number of top features to return for each class.

        Returns:
            dict with 'ai_features' and 'human_features' - lists of (feature, score)
                tuples.
        """
        if self._vectorizer is None or self._binary_classifier is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        # Get feature vector for this text
        x = self._vectorizer.transform([text])

        # Get feature names
        feature_names = self._vectorizer.get_feature_names_out()

        # For GradientBoosting, we can use feature importances combined with the actual
        # feature values to understand which features influenced this prediction
        feature_values = x.toarray()[0]
        feature_importances = self._binary_classifier.feature_importances_

        # Calculate influence: importance * presence in text
        influences = feature_importances * feature_values

        # Get top features that push toward AI (class 0)
        top_ai_indices = np.argsort(influences)[-top_n:][::-1]
        ai_features = [
            (feature_names[i], float(influences[i]))
            for i in top_ai_indices
            if influences[i] > 0
        ]

        # Features that push toward human are those with low influence when present
        # or important features that are absent
        top_human_indices = np.argsort(influences)[:top_n]
        human_features = [
            (feature_names[i], float(feature_importances[i]))
            for i in top_human_indices
            if feature_values[i] > 0
        ]

        return {"ai_features": ai_features, "human_features": human_features}

    def get_top_features(self, top_n: int = 20) -> dict[str, list[tuple[str, float]]]:
        """
        Get the most important features for the model overall.

        Args:
            top_n (int): Number of top features to return.

        Returns:
            dict with feature importance information.
        """
        if self._vectorizer is None or self._binary_classifier is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        feature_names = self._vectorizer.get_feature_names_out()
        importances = self._binary_classifier.feature_importances_

        # Get top features by importance
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [(feature_names[i], float(importances[i])) for i in top_indices]

        return {"top_features": top_features}

    def log_feature_analysis(self, top_n: int = 15) -> None:
        """Log analysis of the most important features."""
        if self._vectorizer is None or self._binary_classifier is None:
            return

        logger.info("\n" + "=" * 60)
        logger.info("MODEL INTERPRETABILITY: Top Discriminative Features")
        logger.info("=" * 60)

        top_features = self.get_top_features(top_n)

        logger.info("\nMost Important Character N-grams:")
        for i, (feature, importance) in enumerate(top_features["top_features"], 1):
            logger.info(f"  {i:2d}. '{feature}' → {importance:.6f}")

        logger.info("=" * 60 + "\n")

    def _model_file_exists(self) -> bool:
        p = self._model_path
        if not p.exists():
            return False
        return p.is_file()

    async def _load_dataset(self) -> RawDataset:
        if not hasattr(self, "_dataset_manager"):
            self._dataset_manager: DatasetManager = FilesystemDatasetManager()
            await self._dataset_manager.create_dataset_if_missing(
                dataset_name=config.main_dataset,
                samples=config.samples,
                subset_split=None,
            )

        return await self._dataset_manager.get_raw_dataset(config.main_dataset)

    async def _train(self) -> float:
        self._binary_classifier = GradientBoostingClassifier(
            n_estimators=100, random_state=0
        )
        dataset = await self._load_dataset()

        # Validate that we have training samples.
        if dataset.training.get_size() == 0:
            logger.error(
                f"Dataset has no training samples. Training set size: "
                f"{dataset.training.get_size()}, "
                f"Validation set size: {dataset.validation.get_size()}, "
                f"Testing set size: {dataset.testing.get_size()}"
            )
            raise ValueError("Dataset has no training samples. Cannot train the model.")

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(
                self._vectoriser_ngram_range_min,
                self._vectoriser_ngram_range_max,
            ),
            min_df=self._vectoriser_min_df,
            max_df=self._vectoriser_max_df,
            sublinear_tf=True,
        )

        # Extract training texts and labels
        train_texts = [sample.text for sample in dataset.training.samples]
        train_labels = [sample.label.value for sample in dataset.training.samples]

        x_train = self._vectorizer.fit_transform(train_texts)
        self._binary_classifier = self._binary_classifier.fit(X=x_train, y=train_labels)

        # Extract validation texts and labels
        val_texts = [sample.text for sample in dataset.validation.samples]
        val_labels = [sample.label.value for sample in dataset.validation.samples]

        x_validation = self._vectorizer.transform(val_texts)

        # Returns the accuracy on a validation set.
        return float(self._binary_classifier.score(X=x_validation, y=val_labels))

    def _load_model(self) -> GradientBoostingClassifier:
        with self._model_path.open("rb") as f:
            model_data = load(f)  # noqa: S301, we have chosen `pickle` due its simplicity.
            self._vectorizer = model_data["vectorizer"]
            self._binary_classifier = model_data["classifier"]
            if self._binary_classifier is None:
                raise ValueError("Failed to load the binary classifier from the file.")
        return self._binary_classifier

    def _save_model(self) -> None:
        with self._model_path.open("wb") as f:
            dump(
                {
                    "classifier": self._binary_classifier,
                    "vectorizer": self._vectorizer,
                },
                f,
                protocol=5,
            )
