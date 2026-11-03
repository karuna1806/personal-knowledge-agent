"""One client, swappable backend (§3.4). Ollama and vLLM both speak the OpenAI
`/v1/chat/completions` protocol, so the backend is a config value."""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from pka.core.metrics import STAGE_LATENCY
from pka.core.telemetry import tracer

Message = dict[str, str]


@dataclass
class StreamStats:
    """Filled in while a stream is consumed; read after it finishes."""

    started: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def ttft_ms(self) -> int | None:
        if self.first_token_at is None:
            return None
        return int((self.first_token_at - self.started) * 1000)

    @property
    def total_ms(self) -> int | None:
        if self.finished_at is None:
            return None
        return int((self.finished_at - self.started) * 1000)

    @property
    def tokens_per_second(self) -> float | None:
        if not (self.completion_tokens and self.first_token_at and self.finished_at):
            return None
        gen_s = self.finished_at - self.first_token_at
        return self.completion_tokens / gen_s if gen_s > 0 else None


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        reasoning_effort: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
    ):
        self._client = AsyncOpenAI(base_url=base_url, api_key="not-needed", timeout=timeout_s)
        self.model = model
        self.base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra: dict[str, Any] = {}
        if reasoning_effort:
            self._extra["reasoning_effort"] = reasoning_effort

    async def stream(
        self,
        messages: list[Message],
        stats: StreamStats | None = None,
        *,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield content deltas. Reasoning/thinking deltas are discarded."""
        stats = stats if stats is not None else StreamStats()
        with tracer.start_as_current_span("llm.stream") as span:
            span.set_attribute("llm.model", self.model)
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                temperature=self._temperature,
                max_tokens=max_tokens or self._max_tokens,
                extra_body=self._extra or None,
            )
            try:
                async for chunk in response:
                    if chunk.usage:
                        stats.prompt_tokens = chunk.usage.prompt_tokens
                        stats.completion_tokens = chunk.usage.completion_tokens
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content
                    if text:
                        if stats.first_token_at is None:
                            stats.first_token_at = time.perf_counter()
                        yield text
            finally:
                await response.close()
                stats.finished_at = time.perf_counter()
                if stats.ttft_ms is not None:
                    span.set_attribute("llm.ttft_ms", stats.ttft_ms)
                if stats.completion_tokens is not None:
                    span.set_attribute("llm.completion_tokens", stats.completion_tokens)

    async def complete(self, messages: list[Message], *, max_tokens: int | None = None) -> str:
        with tracer.start_as_current_span("llm.complete"), STAGE_LATENCY.labels("llm_complete").time():
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=max_tokens or self._max_tokens,
                extra_body=self._extra or None,
            )
        return response.choices[0].message.content or ""

    async def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        *,
        name: str = "output",
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Constrained decoding (§3.7): the server forces output to conform to `schema`
        (llama.cpp GBNF in Ollama, xgrammar/outlines in vLLM). Malformed output becomes
        structurally impossible rather than something we validate-and-retry."""
        with tracer.start_as_current_span("llm.complete_json") as span, STAGE_LATENCY.labels(
            f"llm_json_{name}"
        ).time():
            span.set_attribute("llm.schema", name)
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": schema, "strict": True},
                },
                extra_body=self._extra or None,
            )
        raw = response.choices[0].message.content or ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:  # e.g. truncated at max_tokens
            raise LLMError(f"model returned non-JSON for {name!r}: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"model returned non-object JSON for {name!r}")
        return parsed

    async def ping(self) -> bool:
        try:
            await self._client.models.list(timeout=3)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()
