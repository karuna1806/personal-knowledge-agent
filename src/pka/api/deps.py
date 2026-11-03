"""Dependency injection: app context, auth (single API key), rate limiting."""

import hashlib
import secrets
import time

from fastapi import Depends, HTTPException, Request, status

from pka.api.context import AppContext


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def _presented_key(request: Request) -> str | None:
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


async def require_api_key(request: Request, ctx: AppContext = Depends(get_ctx)) -> None:
    expected = ctx.settings.api_key
    if not expected:  # auth disabled (dev only)
        return
    presented = _presented_key(request)
    if presented is None or not secrets.compare_digest(presented.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key",
                            headers={"WWW-Authenticate": "Bearer"})  # fmt: skip


async def rate_limit(request: Request, ctx: AppContext = Depends(get_ctx)) -> None:
    """Fixed one-minute window per client in Redis. Fails open: if Redis is down we lose
    rate limiting, not availability."""
    limit = ctx.settings.rate_limit_per_minute
    if limit <= 0:
        return
    client = _presented_key(request) or (request.client.host if request.client else "anon")
    window = int(time.time() // 60)
    # Stable across processes (built-in hash() is randomised per interpreter).
    key = f"rl:{hashlib.sha256(client.encode()).hexdigest()[:16]}:{window}"
    try:
        async with ctx.redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, 65)
            count, _ = await pipe.execute()
    except Exception:
        return
    if count > limit:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded",
                            headers={"Retry-After": str(60 - int(time.time()) % 60)})  # fmt: skip
