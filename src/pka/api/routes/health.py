"""/health = the process is alive. /ready = every dependency is reachable (§8)."""

import asyncio

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from pka.api.context import AppContext
from pka.api.deps import get_ctx

router = APIRouter(tags=["ops"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
async def ready(response: Response, ctx: AppContext = Depends(get_ctx)) -> dict:
    async def postgres() -> bool:
        async with ctx.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True

    async def redis() -> bool:
        return bool(await ctx.redis.ping())

    async def rabbitmq() -> bool:
        connection = ctx.broker.connection
        return bool(ctx.broker_ready and connection is not None and not connection.is_closed)

    async def llm() -> bool:
        return await ctx.llm.ping()

    async def check(fn) -> bool:
        try:
            return bool(await asyncio.wait_for(fn(), timeout=3))
        except Exception:
            return False

    names = ("postgres", "redis", "rabbitmq", "llm")
    results = await asyncio.gather(*(check(fn) for fn in (postgres, redis, rabbitmq, llm)))
    checks = dict(zip(names, results, strict=True))
    ok = all(checks.values())
    response.status_code = 200 if ok else 503
    return {
        "status": "ready" if ok else "degraded",
        "checks": checks,
        "tools": sorted(ctx.tools.tools),
        "reranker": getattr(ctx.reranker, "model_name", None),
    }


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
