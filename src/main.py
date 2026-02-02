"""Entry point to the application as a Typer CLI."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.router import lifespan
from src.api.router import router as main_router
from src.configuration import config


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


if __name__ == "__main__":
    run_api()
