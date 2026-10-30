"""The FastAPI application.

Models load in the lifespan, before the first request is served. A readiness
probe that returns 200 while the weights are still loading is worse than no
probe at all: a load balancer would send traffic to a process that cannot
answer it.
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from voicebot import __version__
from voicebot.api.deps import build_services, check_startup
from voicebot.api.routes import router
from voicebot.config import AppConfig, build_config, load_config
from voicebot.errors import ErrorCode, InternalError, VoicebotError, current_call_id
from voicebot.live.routes import router as live_router
from voicebot.messages import say
from voicebot.utils.env import load_env_file
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report

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


def _thread_failed(args: threading.ExceptHookArgs) -> None:
    """A background thread died: say so in the log, with its traceback, and count it.

    Without this an exception in a worker thread goes to stderr, outside the
    log, and nothing counts it -- the thread is simply gone.
    """
    name = args.thread.name if args.thread else "a thread"
    log.error("%s stopped: %s", name, args.exc_type.__name__, exc_info=args.exc_value)
    if isinstance(args.exc_value, Exception):
        report(InternalError(f"{name} stopped", component="thread", cause=args.exc_value))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the engines once, before serving; drain and release them after."""
    # Before anything reads the environment, so a .env beside the project is
    # as good as an export. Anything already exported still wins.
    load_env_file()
    threading.excepthook = _thread_failed

    config_path = os.environ.get("VOICEBOT_CONFIG")
    config: AppConfig = load_config(config_path) if config_path else build_config({})
    # Every reason the service could not work, at once, before anything opens.
    check_startup(config)

    log.info("loading services (config=%s)", config_path or "defaults")
    app.state.services = build_services(config=config)
    # The fallback sentences are recorded in the background: startup does not
    # wait on the voices, and a voice that is down now is tried again later.
    threading.Thread(target=app.state.services.warm, name="warm-voices", daemon=True).start()
    try:
        yield
    finally:
        services, app.state.services = app.state.services, None
        if services is not None:
            # New turns are told the service is restarting; the ones running
            # are given time to finish. Then each part is closed on its own.
            services.drain()
            services.close()
        log.info("services released")


def _safe(error: VoicebotError | None, status: int) -> JSONResponse:
    """A failure as the page may see it: a sentence, a code, and the call -- nothing internal."""
    code = error.code if error is not None else ErrorCode.INTERNAL
    key = (
        "restarting"
        if status == 503 and code is ErrorCode.DATABASE_ERROR
        else (
            "speech_unavailable"
            if error is not None and error.component.startswith("asr")
            else "internal"
        )
    )
    return JSONResponse(
        status_code=status,
        content={
            "detail": say(key, "en"),
            "code": str(code),
            "call_id": (error.call_id if error is not None else None) or current_call_id(),
        },
    )


def _install_handlers(app: FastAPI) -> None:
    @app.exception_handler(VoicebotError)
    async def known(request: Request, exc: VoicebotError) -> JSONResponse:
        report(exc)
        status = 400 if exc.code in (ErrorCode.INVALID_INPUT, ErrorCode.AUTH_FAILURE) else 503
        return _safe(exc, status)

    @app.exception_handler(RequestValidationError)
    async def malformed(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Which fields, never what was in them: a request with a PIN in it is
        # not echoed back, not even to the page that sent it.
        fields = sorted({".".join(str(p) for p in e.get("loc", ())[1:]) for e in exc.errors()})
        log.info("refused a malformed request to %s: %s", request.url.path, fields)
        return JSONResponse(
            status_code=422,
            content={
                "detail": "The request was not in the expected form.",
                "code": str(ErrorCode.INVALID_INPUT),
                "fields": fields,
            },
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled %s on %s", type(exc).__name__, request.url.path, exc_info=exc)
        report(InternalError(f"unhandled {type(exc).__name__}", component="api", cause=exc))
        return _safe(None, 500)


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

    _install_handlers(app)
    app.include_router(router)
    app.include_router(live_router)

    # Mounted last and at the root, so every API path above still wins. The
    # frontend then shares an origin with the API, which means the page can
    # use relative URLs and the browser will grant it the microphone.
    if STATIC_DIR.is_dir():
        app.mount("/", RevalidatedStatic(directory=STATIC_DIR, html=True), name="ui")
    else:  # pragma: no cover - only when the package is installed without assets
        log.warning("no static directory at %s; the API is up but the UI is not", STATIC_DIR)

    return app


app = create_app()
