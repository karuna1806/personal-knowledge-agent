"""MCP client: connect to configured servers over stdio, discover their tools at runtime,
and call them with a timeout and an output cap."""

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass

import mcp

from pka.core.metrics import TOOL_CALLS
from pka.core.telemetry import tracer
from pka.mcp.registry import ServerConfig

log = logging.getLogger(__name__)

SEPARATOR = "__"


@dataclass
class ToolSpec:
    name: str  # qualified: "<server>__<tool>"
    server: str
    tool: str
    description: str
    input_schema: dict

    def summary(self) -> str:
        props = self.input_schema.get("properties", {})
        args = ", ".join(props) if props else "no arguments"
        first_line = (self.description or "").strip().split("\n")[0]
        return f"{self.name}({args}): {first_line}"


@dataclass
class ToolResult:
    text: str
    is_error: bool


class MCPManager:
    def __init__(self, *, timeout_s: float = 20.0, max_chars: int = 6000):
        self._timeout_s = timeout_s
        self._max_chars = max_chars
        self._stack = AsyncExitStack()
        self._clients: dict[str, mcp.Client] = {}
        self.tools: dict[str, ToolSpec] = {}

    async def start(self, servers: list[ServerConfig]) -> None:
        for server in servers:
            if not server.enabled:
                continue
            try:
                await asyncio.wait_for(self._connect(server), timeout=30)
            except Exception as exc:
                # A broken tool server must not take the agent down; its tools just vanish.
                log.warning("MCP server %s unavailable: %s", server.name, exc)
        log.info("MCP tools discovered", extra={"tools": sorted(self.tools)})

    async def _connect(self, server: ServerConfig) -> None:
        params = mcp.StdioServerParameters(
            command=server.command,
            args=server.args,
            env={**os.environ, **server.env},
        )
        client = mcp.Client(params, read_timeout_seconds=self._timeout_s)
        await self._stack.enter_async_context(client)
        listed = await client.list_tools()
        self._clients[server.name] = client
        for tool in listed.tools:
            if server.allowed_tools is not None and tool.name not in server.allowed_tools:
                continue
            qualified = f"{server.name}{SEPARATOR}{tool.name}"
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
            self.tools[qualified] = ToolSpec(qualified, server.name, tool.name, tool.description or "", schema)

    def tool_lines(self) -> list[str]:
        return [spec.summary() for spec in self.tools.values()]

    async def call(self, name: str, arguments: dict) -> ToolResult:
        spec = self.tools.get(name)
        if spec is None:
            TOOL_CALLS.labels(name, "unknown").inc()
            return ToolResult(f"unknown tool {name!r}", True)
        client = self._clients[spec.server]
        with tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("tool.name", name)
            try:
                result = await asyncio.wait_for(client.call_tool(spec.tool, arguments), timeout=self._timeout_s)
            except Exception as exc:
                TOOL_CALLS.labels(name, "error").inc()
                return ToolResult(f"tool call failed: {type(exc).__name__}: {exc}", True)
        is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
        parts = []
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else f"[{getattr(block, 'type', 'content')} omitted]")
        output = "\n".join(parts).strip() or "(no output)"
        if len(output) > self._max_chars:
            output = output[: self._max_chars] + f"\n… [truncated, {len(output) - self._max_chars} more chars]"
        TOOL_CALLS.labels(name, "error" if is_error else "ok").inc()
        return ToolResult(output, is_error)

    async def close(self) -> None:
        try:
            await self._stack.aclose()
        except Exception as exc:  # server processes may already be gone at shutdown
            log.debug("error closing MCP clients: %s", exc)
        self._clients.clear()
        self.tools.clear()
