"""FastAPI gateway: auth · validation · rate limit · streaming. No retrieval logic here."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from pka.api.context import build_context
from pka.api.middleware import RequestIdMiddleware
from pka.api.routes import admin, corpus, health, ingest, memory, query
from pka.core.config import get_settings
from pka.core.logging import configure_logging
from pka.core.telemetry import configure_tracing

log = logging.getLogger("pka.api")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    configure_tracing(settings.service_name, settings.otel_exporter_otlp_endpoint)
    if not settings.api_key:
        log.warning("API_KEY is empty: authentication is DISABLED")
    ctx = await build_context(settings)
    app.state.ctx = ctx
    log.info(
        "api ready",
        extra={"llm": f"{ctx.llm.model} @ {ctx.llm.base_url}", "embed_model": ctx.embedder.model_name,
               "reranker": getattr(ctx.reranker, "model_name", None), "tools": sorted(ctx.tools.tools)},
    )  # fmt: skip
    try:
        yield
    finally:
        await ctx.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Personal Knowledge Agent",
        version="0.1.0",
        description="Self-hosted RAG over my notes, code and documents, with a persistent memory layer.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    for module in (health, query, ingest, corpus, memory, admin):
        app.include_router(module.router)
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app


app = create_app()
