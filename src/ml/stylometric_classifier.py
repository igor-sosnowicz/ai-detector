"""Module with a text binary classifier for stylometric features."""

from pathlib import Path
from pickle import dump, load

from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.configuration import config
from src.data_models import RawDataset
from src.training.dataset import DatasetManager, FilesystemDatasetManager


class StylometricClassifier:
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

    async def test(self) -> float:
        """Return accuracy on the testing set."""
        if self._vectorizer is None or self._binary_classifier is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")

        dataset = await self._load_dataset()
        if dataset.testing.get_size() == 0:
            raise ValueError("Dataset has no testing samples. Cannot test the model.")

        test_texts = [sample.text for sample in dataset.testing.samples]
        test_labels = [sample.label.value for sample in dataset.testing.samples]

        x_test = self._vectorizer.transform(test_texts)
        return float(self._binary_classifier.score(X=x_test, y=test_labels))

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
        # 1.0 - AI, 0.0 - human.
        if self._vectorizer is None or self._binary_classifier is None:
            raise RuntimeError(
                f"Model not prepared. Call `{self.prepare_model.__name__}` first."
            )
        x = self._vectorizer.transform([text])
        probabilities = self._binary_classifier.predict_proba(X=x)
        return float(probabilities[0, 1])  # Get probability of class 1 (AI)

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
            ngram_range=(2, 5),
            min_df=2,
            max_df=0.95,
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
