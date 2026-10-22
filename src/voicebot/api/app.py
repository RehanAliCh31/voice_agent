"""The FastAPI application.

Models load in the lifespan, before the first request is served. A readiness
probe that returns 200 while the weights are still loading is worse than no
probe at all: a load balancer would send traffic to a process that cannot
answer it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from voicebot import __version__
from voicebot.api.deps import build_services
from voicebot.api.routes import router
from voicebot.config import AppConfig, load_config
from voicebot.utils.logging import get_logger

log = get_logger("api.app")

STATIC_DIR = Path(__file__).parent / "static"


class RevalidatedStatic(StaticFiles):
    """Static files a browser must check before reusing.

    StaticFiles sends an ETag but no Cache-Control, so a browser is free to
    apply heuristic caching and serve a stale script without asking. For a
    page whose JavaScript changes with the service, that means someone runs
    yesterday's code and reports that a fix did not work.

    `no-cache` does not forbid caching -- it requires revalidation, which the
    existing ETag answers with a 304 and no body.
    """

    async def get_response(self, path: str, scope: Scope):  # type: ignore[no-untyped-def]
        response = await super().get_response(path, scope)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


DESCRIPTION = """
A cascaded speech-to-speech banking assistant.

`/v1/converse` runs a whole turn: audio in, audio out. The individual stages
are also exposed, so a client can inspect any one of them in isolation.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the engines once, before serving."""
    config_path = os.environ.get("VOICEBOT_CONFIG")
    config: AppConfig = load_config(config_path) if config_path else AppConfig()

    log.info("loading services (config=%s)", config_path or "defaults")
    app.state.services = build_services(config=config)
    try:
        yield
    finally:
        app.state.services = None
        log.info("services released")


def create_app() -> FastAPI:
    """Build the application. A factory, so tests can make an isolated one."""
    app = FastAPI(
        title="Voice Bot",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    # Kept for the case where the page is hosted apart from the API. Serving
    # both from here makes them same-origin, so this is belt and braces.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # Mounted last and at the root, so every API path above still wins. The
    # frontend then shares an origin with the API, which means the page can
    # use relative URLs and the browser will grant it the microphone.
    if STATIC_DIR.is_dir():
        app.mount("/", RevalidatedStatic(directory=STATIC_DIR, html=True), name="ui")
    else:  # pragma: no cover - only when the package is installed without assets
        log.warning("no static directory at %s; the API is up but the UI is not", STATIC_DIR)

    return app


app = create_app()
