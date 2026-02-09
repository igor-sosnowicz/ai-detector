"""Entry point to the application as a Typer CLI."""

import asyncio

from loguru import logger
from typer import Typer

from src.api.utils import get_ip_address_or_raise
from src.configuration import config

app = Typer(no_args_is_help=True)

description = """
**Checktica** offers the API for checking if a text was written by AI or a human.

## Why Checktica?

- **99%+ accuracy** in AI text detection task.
- **Free** to use. You pay exactly 0$.
- **No API key** required.
- **No text length limit**. Send as long texts as you need.
"""


@app.command("api")
def run_api() -> None:
    """Start up the backed sharing the Web API."""
    import uvicorn
    from api_analytics.fastapi import Analytics, Config
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    from src.api.router import lifespan
    from src.api.router import router as main_router

    fastapi_app = FastAPI(
        title="Checktica",
        summary="Checktica API detects AI-written texts.",
        description=description,
        lifespan=lifespan,
        docs_url="/v1/docs",
        openapi_url="/v1/openapi.json",
        redoc_url="/v1/redoc",
    )

    # CORS is handled by nginx - no middleware needed here.

    # API usage analytics.
    analytics_config = Config()
    analytics_config.get_ip_address = get_ip_address_or_raise

    fastapi_app.add_middleware(
        Analytics,
        api_key=config.apianalyticsdev_api_key,
        config=analytics_config,
    )

    @fastapi_app.get("/")
    async def root() -> RedirectResponse:
        """Redirect root to docs."""
        return RedirectResponse(url="/v1/docs")

    @fastapi_app.get("/v1")
    async def root_v1() -> RedirectResponse:
        """Redirect /v1 to docs."""
        return RedirectResponse(url="/v1/docs")

    fastapi_app.include_router(main_router, prefix="/v1")
    uvicorn.run(fastapi_app, host=config.api_host, port=config.api_port)


@app.command("optimise")
def optimise_detectors() -> None:
    """Optimise parameters of detectors."""
    from src.detection.compression_based import CompressionBasedDetector
    from src.detection.heuristics import Heuristics
    from src.detection.spelling import SpellingDetector
    from src.ml.stylometric_classifier import StylometricClassifier
    from src.training.evaluation import Evaluator
    from src.training.tuning import optimise

    english_heuristics = Heuristics(language="english")
    perplexity_detector = CompressionBasedDetector()
    stylometric_classifier = StylometricClassifier()
    english_spelling = SpellingDetector()

    asyncio.run(stylometric_classifier.prepare_model())

    for detector in {
        english_heuristics,
        perplexity_detector,
        # stylometric_classifier, # It has persistent weights that are not yet supported
        english_spelling,
    }:
        optimal_parameters = optimise(detector=detector)
        logger.info(
            f"Setting optimal parameters for {type(detector).__qualname__}: "
            f"{optimal_parameters}"
        )

        detector.set_tunable_attributes(optimal_parameters)
        detector.save()

    evaluator = Evaluator()
    evaluator.add_detector(english_heuristics)
    evaluator.add_detector(perplexity_detector)
    evaluator.add_detector(stylometric_classifier)
    evaluator.add_detector(english_spelling)

    results = asyncio.run(evaluator.evaluate(subset="testing"))
    logger.info("Results of the evaluation are the following:")
    for detector_name, evaluation in results.items():
        logger.info(f"\nDetector: {detector_name}\n{evaluation}")


if __name__ == "__main__":
    app()
