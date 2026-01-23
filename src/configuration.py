"""The configuration module."""

import tomllib
from pathlib import Path

from pydantic import BaseModel


class Configuration(BaseModel):
    """Configuration of the application."""

    project_name: str


def load_configuration(
    configuration_file: Path = Path("config.toml"),
) -> Configuration:
    """Load configuration from the configuration file."""
    with configuration_file.open("rb") as f:
        settings = tomllib.load(f)
    return Configuration(**settings)


config = load_configuration()
