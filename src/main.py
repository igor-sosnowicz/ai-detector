"""Entry point to the application as a Typer CLI."""

import asyncio

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger
from typer import Typer

from src.api.router import lifespan
from src.api.router import router as main_router
from src.configuration import config
from src.training.evaluation import Evaluator
from src.training.tuning import optimise

app = Typer(no_args_is_help=True)


@app.command("api")
def run_api() -> None:
    """Start up the backed sharing the Web API."""
    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> RedirectResponse:
        """Redirect root to docs."""
        return RedirectResponse(url="/docs")

    app.include_router(main_router)
    uvicorn.run(app, host=config.api_host, port=config.api_port)


@app.command("optimise")
def optimise_detectors() -> None:
    """Optimise parameters of detectors."""
    from src.detection.compression_based import CompressionBasedDetector
    from src.detection.heuristics import Heuristics
    from src.detection.spelling import SpellingDetector
    from src.ml.stylometric_classifier import StylometricClassifier

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
