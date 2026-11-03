"""Everything the gateway holds for its lifetime, built once in the lifespan."""

import asyncio
import logging
from dataclasses import dataclass, field

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from pka.agent.core import Agent
from pka.agent.reranker import CrossEncoderReranker, PassthroughReranker, Reranker
from pka.agent.retriever import Retriever
from pka.core.config import Settings
from pka.core.metrics import DLQ_DEPTH, QUEUE_DEPTH, update_accelerator_memory
from pka.db.session import make_engine, make_sessionmaker
from pka.ingest.embed import Embedder, make_encoder
from pka.ingest.tokens import get_token_counter
from pka.llm.client import LLMClient
from pka.mcp.client import MCPManager
from pka.mcp.registry import load_registry
from pka.queue.broker import DLQ, Broker

log = logging.getLogger(__name__)


@dataclass
class AppContext:
    settings: Settings
    engine: AsyncEngine
    sessionmaker: async_sessionmaker[AsyncSession]
    redis: Redis
    llm: LLMClient
    embedder: Embedder
    reranker: Reranker
    broker: Broker
    tools: MCPManager
    agent: Agent
    broker_ready: bool = False
    _tasks: list[asyncio.Task] = field(default_factory=list)

    async def publish(self, routing_key: str, payload: dict) -> None:
        if not self.broker_ready:
            raise RuntimeError("message broker unavailable")
        await self.broker.publish(routing_key, payload)

    async def _connect_broker(self) -> None:
        try:
            await self.broker.connect()
            self.broker_ready = True
            log.info("broker connected")
        except Exception as exc:
            log.warning("broker unavailable (%s); will retry", exc)

    async def _monitor(self) -> None:
        """Reconnect the broker if startup missed it; export queue/DLQ depth gauges."""
        while True:
            if not self.broker_ready:
                await self._connect_broker()
            if self.broker_ready:
                try:
                    for name, depth in (await self.broker.queue_depths()).items():
                        QUEUE_DEPTH.labels(name).set(depth)
                        if name == DLQ:
                            DLQ_DEPTH.set(depth)
                except Exception as exc:
                    log.debug("queue depth poll failed: %s", exc)
            update_accelerator_memory()
            await asyncio.sleep(15)

    def start_background(self) -> None:
        self._tasks.append(asyncio.create_task(self._monitor(), name="broker-monitor"))

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.tools.close()
        await self.broker.close()
        await self.llm.close()
        await self.redis.aclose()
        await self.engine.dispose()


async def build_context(settings: Settings) -> AppContext:
    engine = make_engine(settings.database_url)
    sessionmaker = make_sessionmaker(engine)
    redis = Redis.from_url(settings.redis_url, socket_timeout=2, socket_connect_timeout=2)
    encoder = await asyncio.to_thread(
        make_encoder,
        settings.embed_model,
        settings.embed_dim,
        device=settings.embed_device,
        query_prompt=settings.embed_query_prompt,
        batch_size=settings.embed_batch_size,
    )
    embedder = Embedder(encoder, redis, settings.embed_cache_ttl_s)
    reranker: Reranker
    if settings.rerank_enabled:
        reranker = await asyncio.to_thread(
            CrossEncoderReranker, settings.rerank_model, settings.embed_device, settings.rerank_max_length
        )
    else:
        reranker = PassthroughReranker()
    llm = LLMClient(
        settings.llm_base_url,
        settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
    )
    tools = MCPManager(timeout_s=settings.mcp_tool_timeout_s, max_chars=settings.mcp_result_max_chars)
    if settings.mcp_enabled:
        await tools.start(load_registry(settings.mcp_config))
    broker = Broker(settings.amqp_url, settings.queue_retry_delay_ms)

    ctx = AppContext(
        settings=settings, engine=engine, sessionmaker=sessionmaker, redis=redis, llm=llm,
        embedder=embedder, reranker=reranker, broker=broker, tools=tools,
        agent=None,  # type: ignore[arg-type]  — set below; needs ctx.publish
    )  # fmt: skip
    ctx.agent = Agent(
        settings=settings,
        sessionmaker=sessionmaker,
        redis=redis,
        llm=llm,
        embedder=embedder,
        retriever=Retriever(sessionmaker, embedder, settings),
        reranker=reranker,
        count_tokens=get_token_counter(settings.embed_model),
        tools=tools,
        publish=ctx.publish,
    )
    await ctx._connect_broker()
    ctx.start_background()
    return ctx
