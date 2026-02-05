"""Module with generators of new dataset samples."""

import asyncio
import random
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from typing import override

import httpx
import pandas as pd
from cerebras.cloud.sdk import RateLimitError
from loguru import logger
from tqdm import tqdm

from src.configuration import config
from src.data_models import Sample, SampleType
from src.training.prompt_manager import SQLitePromptManger
from src.training.sanitiser import sanitise
from src.utils import LLM


class SampleGenerator(ABC):
    """An interface for a sample generator."""

    @abstractmethod
    async def generate_samples(
        self, number: int, *, shuffled: bool = True
    ) -> list[Sample]:
        """
        Generate samples of human- and LLM-written texts.

        Args:
            number (int): A number of samples to generated.
            shuffled (bool, optional): Whether samples should be shuffled
                after generation. Defaults to True.

        Returns:
            list[Sample]: A list of samples.
        """


class LLMSampleGenerator(SampleGenerator):
    """LLM samples generator."""

    def __init__(self) -> None:
        """Set a path to Parquet dataset with human samples and download timeout."""
        self._parquet_file = Path("./data/parquet/ai_vs_llm.parquet")
        self._timeout_seconds = 600

    def _get_exactly_n_prompts(self, n: int) -> Queue[tuple[str, str]]:
        # Generate a requested number of prompts.
        with SQLitePromptManger() as prompt_manager:
            prompt_queue = Queue(maxsize=n)
            while True:
                for uuid, prompt in prompt_manager.get_prompts(
                    uuids=None, limit=n
                ).items():
                    if prompt_queue.full():
                        return prompt_queue
                    prompt_queue.put((uuid, prompt))

    async def _download_dataset_if_missing(self) -> Path:
        if self._parquet_file.exists():
            return self._parquet_file

        url = "https://huggingface.co/datasets/gsingh1-py/train/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true"
        self._parquet_file.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(
            timeout=self._timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            self._parquet_file.write_bytes(response.content)

        return self._parquet_file

    @override
    async def generate_samples(
        self, number: int, *, shuffled: bool = True
    ) -> list[Sample]:
        # It generates only LLM samples, no human samples are taken.
        # Equal number of samples from each LLM.
        # Ignore `shuffled`.
        system_prompt = (
            "You are a helpful assistant. Follow instructions provided by the user."
        )
        prompt_queue = self._get_exactly_n_prompts(n=number)
        models = len(config.llms_for_samples)
        tasks = []
        used_models = []
        used_prompt_uuids = []

        for model in config.llms_for_samples:
            for _ in range(number // models):
                llm = LLM(model=model, system_prompt=system_prompt)
                prompt_uuid, prompt = prompt_queue.get()
                tasks.append(
                    llm.get_response(
                        prompt=prompt,
                    )
                )
                used_models.append(model)
                used_prompt_uuids.append(prompt_uuid)

        # Wrap tasks with progress tracking.
        pbar = tqdm(
            total=len(tasks), desc="Generating LLM samples (API)", unit="sample"
        )

        async def track_task(task: asyncio.Future) -> str | Exception:
            try:
                result = await task
            except (TimeoutError, httpx.HTTPError, ValueError, RateLimitError) as e:
                pbar.update(1)
                # Return exception instead of raising.
                return e
            else:
                pbar.update(1)
                return result

        tracked_tasks = [track_task(task) for task in tasks]
        llm_responses = await asyncio.gather(*tracked_tasks, return_exceptions=False)
        pbar.close()

        # Separate successful and failed responses.
        samples = []
        failed_indices = []
        for i, (prompt_uuid, model, response) in enumerate(
            zip(used_prompt_uuids, used_models, llm_responses, strict=True)
        ):
            if isinstance(response, Exception):
                failed_indices.append(i)
            else:
                samples.append(
                    Sample(
                        text=sanitise(response),
                        label=SampleType.FULLY_LLM_WRITTEN,
                        author=model,
                        prompt_uuid=prompt_uuid,
                    )
                )

        return samples

    async def generate_samples_from_parquet(
        self, number: int, *, shuffled: bool = True
    ) -> list[Sample]:
        """Generate LLM samples from a Parquet file."""
        dataset_path = await self._download_dataset_if_missing()
        df = pd.read_parquet(dataset_path)

        llm_response_cols = [
            "gemma-2-9b",
            "mistral-7B",
            "qwen-2-72B",
            "llama-8B",
            "accounts/yi-01-ai/models/yi-large",
            "GPT_4-o",
        ]

        if not llm_response_cols:
            raise ValueError("No LLM response columns found in Parquet file.")

        llm_response_col = llm_response_cols[0]

        # Filter valid rows
        df = df[df[llm_response_col].notna() & df["prompt"].notna()]
        df = df[df[llm_response_col].apply(lambda x: isinstance(x, str) and len(x) > 0)]

        if len(df) < number:
            raise ValueError(
                f"Parquet file has {len(df)} valid LLM samples but {number} were "
                "requested."
            )

        samples = []

        with SQLitePromptManger() as prompt_manager:
            existing_prompts = set(prompt_manager.get_prompts(uuids=None).values())
            filtered_df = df[~df["prompt"].isin(existing_prompts)]

            if filtered_df.empty:
                logger.warning("No new prompts available for LLM Parquet fallback.")
                filtered_df = df

            sample_count = min(number, len(filtered_df))
            sampled_df = filtered_df[["prompt", llm_response_col]].sample(
                n=sample_count
            )

            for _, row in tqdm(
                sampled_df.iterrows(),
                total=len(sampled_df),
                desc="Generating LLM samples (Parquet)",
                unit="sample",
            ):
                prompt_uuid = prompt_manager.add_prompt(row["prompt"])
                samples.append(
                    Sample(
                        author="LLM from parquet fallback",
                        label=SampleType.FULLY_LLM_WRITTEN,
                        prompt_uuid=prompt_uuid,
                        text=sanitise(self._normalize_text(row[llm_response_col])),
                    )
                )

        if len(samples) < number:
            logger.warning(
                f"Could only generate {len(samples)} LLM samples from Parquet out of "
                f"{number} requested (not enough new prompts available)."
            )

        if shuffled:
            random.shuffle(samples)

        return samples

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by collapsing whitespace."""
        text = text.replace("\n", " ").replace("\t", " ")
        return " ".join(text.split())


class HumanSampleGenerator(SampleGenerator):
    """Sample generator from human-created texts."""

    def __init__(self) -> None:
        """Set Parquet file location and dataset download timeout."""
        self._parquet_file = Path("./data/parquet/ai_vs_llm.parquet")
        self._timeout_seconds = 600

    async def _download_dataset_if_missing(self) -> Path:
        if self._parquet_file.exists():
            return self._parquet_file

        url = "https://huggingface.co/datasets/gsingh1-py/train/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet?download=true"
        self._parquet_file.parent.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(
            timeout=self._timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            self._parquet_file.write_bytes(response.content)

        return self._parquet_file

    @override
    async def generate_samples(
        self, number: int, *, shuffled: bool = True
    ) -> list[Sample]:
        dataset_path = await self._download_dataset_if_missing()

        df = pd.read_parquet(dataset_path)

        # Filter out rows with missing or non-string values
        df = df[df["Human_story"].notna() & df["prompt"].notna()]
        df = df[df["Human_story"].apply(lambda x: isinstance(x, str) and len(x) > 0)]

        if len(df) < number:
            raise ValueError(
                f"Dataset has {len(df)} valid samples but {number} were requested."
            )

        samples = []

        with SQLitePromptManger() as prompt_manager:
            existing_prompts = set(prompt_manager.get_prompts(uuids=None).values())
            filtered_df = df[~df["prompt"].isin(existing_prompts)]

            if filtered_df.empty:
                logger.warning("No new prompts available for human samples.")
                filtered_df = df

            sample_count = min(number, len(filtered_df))
            sampled_df = filtered_df[["prompt", "Human_story"]].sample(n=sample_count)

            for _, row in tqdm(
                sampled_df.iterrows(),
                total=len(sampled_df),
                desc="Generating human samples",
                unit="sample",
            ):
                prompt_uuid = prompt_manager.add_prompt(row["prompt"])
                samples.append(
                    Sample(
                        author="A human writer from: https://huggingface.co/datasets/gsingh1-py",
                        label=SampleType.FULLY_HUMAN_WRITTEN,
                        prompt_uuid=prompt_uuid,
                        text=self._normalize_text(row["Human_story"]),
                    )
                )

        if len(samples) < number:
            logger.warning(
                f"Could only generate {len(samples)} human samples out of {number} "
                "requested (not enough new prompts available)."
            )

        if shuffled:
            random.shuffle(samples)

        return samples

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by collapsing whitespace."""
        # Replace newlines and tabs with spaces.
        text = text.replace("\n", " ").replace("\t", " ")
        return " ".join(text.split())


class MixedSampleGenerator(SampleGenerator):
    """Sample generator for different types of samples."""

    def __init__(self) -> None:
        """Initialise sub-generators generating a single sample type."""
        self._human_generator = HumanSampleGenerator()
        self._llm_generator = LLMSampleGenerator()

    @override
    async def generate_samples(
        self, number: int, *, shuffled: bool = True
    ) -> list[Sample]:
        llm_samples_count = number // 2
        human_samples_count = number - llm_samples_count

        logger.info(
            f"\nGenerating {number} samples ({human_samples_count} human, "
            f"{llm_samples_count} LLM)..."
        )
        human_samples = await self._human_generator.generate_samples(
            human_samples_count
        )

        # Generate LLM samples from APIs
        llm_samples = await self._llm_generator.generate_samples(llm_samples_count)

        # Check if we got fewer samples than requested (due to rate limits or failures)
        missing_count = llm_samples_count - len(llm_samples)
        if missing_count > 0:
            logger.warning(
                f"Only generated {len(llm_samples)}/{llm_samples_count} LLM samples "
                f"via API. Attempting to fill {missing_count} missing samples from "
                "Parquet fallback..."
            )
            try:
                parquet_samples = (
                    await self._llm_generator.generate_samples_from_parquet(
                        missing_count
                    )
                )
                llm_samples.extend(parquet_samples)
                logger.info(
                    f"Successfully filled {len(parquet_samples)} missing samples from "
                    "Parquet fallback."
                )
            except (ValueError, FileNotFoundError, OSError) as parquet_error:
                logger.warning(
                    f"Failed to generate missing samples from Parquet "
                    f"({type(parquet_error).__name__}): {parquet_error}. "
                    f"Generating {missing_count} additional human samples instead."
                )
                # When parquet fails, fill the gap with human samples
                additional_human_samples = await self._human_generator.generate_samples(
                    missing_count
                )
                human_samples.extend(additional_human_samples)

        # Check if we got partial results from API and need to fill the gap
        if len(llm_samples) < llm_samples_count:
            missing_count = llm_samples_count - len(llm_samples)
            logger.info(
                f"Received {len(llm_samples)} LLM samples from API, need "
                f"{missing_count} more. Using Parquet fallback for remaining samples..."
            )
            try:
                fallback_samples = (
                    await self._llm_generator.generate_samples_from_parquet(
                        missing_count
                    )
                )
                llm_samples.extend(fallback_samples)
                logger.info(
                    f"Successfully filled gap with {len(fallback_samples)} samples from"
                    " Parquet fallback"
                )
            except (ValueError, FileNotFoundError, OSError) as fallback_error:
                logger.warning(
                    f"Failed to fill gap from Parquet ({type(fallback_error).__name__})"
                    f": {fallback_error}. Generating additional human samples instead."
                )
                # When Parquet fallback fails, use human samples
                additional_human_samples = await self._human_generator.generate_samples(
                    missing_count
                )
                human_samples.extend(additional_human_samples)

        samples = human_samples + llm_samples
        if not shuffled:
            return samples

        random.shuffle(samples)
        return samples
