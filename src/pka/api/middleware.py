"""Pure-ASGI request-id middleware. (Pure ASGI rather than BaseHTTPMiddleware so the
contextvar is set in the same context that iterates a streaming response body.)"""

import logging
import re
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from pka.core.logging import new_request_id, request_id_var

log = logging.getLogger("pka.access")
_VALID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID.match(incoming) else new_request_id()
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope.get("path", "")
            if path not in ("/health", "/metrics"):
                log.info(
                    "%s %s %s", scope.get("method"), path, status_code,
                    extra={"status": status_code, "duration_ms": int((time.perf_counter() - started) * 1000)},
                )  # fmt: skip
            request_id_var.reset(token)
