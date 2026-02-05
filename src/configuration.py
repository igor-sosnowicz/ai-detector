"""The configuration module."""

import tomllib
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Configuration of the application."""

    project_name: str = "AI Detector"

    api_host: str = "0.0.0.0"  # noqa: S104, it is required for Docker deployment.
    api_port: int = 7123
    api_max_requests_per_interval: int = 1
    api_rate_limiter_interval: timedelta = timedelta(seconds=1)
    cors_origins: list[str] = ["*"]

    suspicious_phrases_file: Path = Path("./data/suspicious_phrases.txt")
    heuristic_threshold: float = 0.65
    stylometric_classifier_path: Path = Path("./data/stylometric_classifier.pickle")

    open_router_api_key: str
    cerebras_api_key: str

    llms_for_samples: list[str] = [
        "gpt-oss-120b",
        "llama3.1-8b",
        "llama-3.3-70b",
        "qwen-3-32b",
    ]

    sqlite_database: Path = Path("./data/prompts.db")
    datasets_directory: Path = Path("./data/datasets")
    llm_cache_directory: Path = Path(".cache/llm")
    nltk_missing_words: Path = Path("./data/unknown_words.txt")

    main_dataset: str = "main"
    samples: int = Field(100, ge=20)


def load_configuration(
    configuration_file: Path = Path("config.toml"),
) -> Configuration:
    """Load configuration from the configuration file."""
    with configuration_file.open("rb") as f:
        settings = tomllib.load(f)
    return Configuration(**settings)


config = load_configuration()
