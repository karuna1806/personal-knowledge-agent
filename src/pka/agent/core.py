"""The agent: route → (retrieve+rerank | bounded tools | direct) → prompt → stream → persist.

No FastAPI imports here (§7): the core emits plain events and is testable without a server.
"""

import hashlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from redis.asyncio import Redis
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pka.agent.citations import extract_citations
from pka.agent.prompt import RULES_DIRECT, RULES_GROUNDED, RULES_TOOL, Source, build_prompt
from pka.agent.reranker import Reranker
from pka.agent.retriever import Retriever, SearchFilters
from pka.agent.router import RouteDecision, route_query
from pka.agent.state import QueryCache, SessionStore
from pka.agent.tools import run_tool_loop
from pka.core.config import Settings
from pka.core.logging import request_id_var
from pka.core.metrics import QUERY_LATENCY, TOKENS_PER_SECOND, TTFT
from pka.core.telemetry import tracer
from pka.db.models import Citation, Conversation, Message, QueryLog
from pka.ingest.embed import Embedder
from pka.ingest.tokens import TokenCounter
from pka.llm.client import LLMClient, StreamStats
from pka.llm.priority import interactive
from pka.mcp.client import MCPManager
from pka.memory.retrieve import retrieve_memories
from pka.queue.broker import MEMORY_EXTRACT

log = logging.getLogger(__name__)

MODES = ("auto", "retrieve", "tool", "direct")


@dataclass
class QueryOptions:
    query: str
    conversation_id: uuid.UUID
    mode: str = "auto"
    filters: SearchFilters | None = None
    use_memory: bool = True
    use_cache: bool = True


@dataclass
class AgentEvent:
    type: str  # meta | sources | token | done | error
    data: dict = field(default_factory=dict)


class UnknownConversation(LookupError):
    pass


def _source_payload(source: Source) -> dict:
    return {
        "n": source.n,
        "kind": source.kind,
        "label": source.label,
        "chunk_id": source.chunk_id,
        "score": round(source.score, 4) if source.score is not None else None,
        "source_uri": source.source_uri,
        "title": source.title,
        "snippet": source.content[:400],
    }


class Agent:
    def __init__(
        self,
        *,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        redis: Redis | None,
        llm: LLMClient,
        embedder: Embedder,
        retriever: Retriever,
        reranker: Reranker,
        count_tokens: TokenCounter,
        tools: MCPManager | None = None,
        publish: Callable[[str, dict], Awaitable[None]] | None = None,
    ):
        self.settings = settings
        self._sessionmaker = sessionmaker
        self._redis = redis
        self.llm = llm
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.tools = tools
        self._publish = publish
        self._count = count_tokens
        self.sessions = SessionStore(redis, sessionmaker, turns=settings.session_turns, ttl_s=settings.session_ttl_s)
        # Anything that changes an answer for the same query must be in the cache key. Corpus
        # and memory changes are covered by version counters; this covers config and prompts.
        prompt_hash = hashlib.sha256((RULES_GROUNDED + RULES_TOOL + RULES_DIRECT).encode()).hexdigest()[:12]
        fingerprint = "|".join(
            str(x) for x in (llm.model, settings.retrieval_mode, getattr(reranker, "model_name", "?"),
                             settings.final_k, embedder.model_name, settings.memory_enabled, prompt_hash)  # fmt: skip
        )
        self.cache = QueryCache(redis, settings.query_cache_ttl_s, fingerprint)

    # ── conversation bookkeeping ──────────────────────────────────
    async def ensure_conversation(self, conversation_id: uuid.UUID | None) -> uuid.UUID:
        async with self._sessionmaker() as session, session.begin():
            if conversation_id is None:
                return await session.scalar(insert(Conversation).values().returning(Conversation.id))
            exists = await session.scalar(select(Conversation.id).where(Conversation.id == conversation_id))
            if exists is None:
                raise UnknownConversation(str(conversation_id))
            return conversation_id

    def tool_lines(self) -> list[str]:
        if not (self.tools and self.settings.mcp_enabled):
            return []
        return self.tools.tool_lines()

    # ── the request path ─────────────────────────────────────────
    async def run(self, opts: QueryOptions) -> AsyncIterator[AgentEvent]:
        # Background LLM work (memory extraction) waits while this query is in flight.
        async with interactive(self._redis, request_id_var.get() + uuid.uuid4().hex[:6]):
            async for event in self._run(opts):
                yield event

    async def _run(self, opts: QueryOptions) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        timings: dict[str, int | None] = {"retrieval_ms": None, "rerank_ms": None, "llm_ms": None, "ttft_ms": None}
        route = "unknown"
        cache_hit = False
        chunks_retrieved = 0
        stats = StreamStats()
        try:
            with tracer.start_as_current_span("agent.run") as span:
                history = await self.sessions.get(opts.conversation_id)
                filtered = opts.filters is not None and not opts.filters.is_empty()
                cacheable = opts.use_cache and not history and opts.mode in ("auto", "retrieve") and not filtered

                if cacheable and (cached := await self.cache.get(opts.query)):
                    route, cache_hit = "retrieve", True
                    async for event in self._replay_cached(opts, cached, started):
                        if event.type == "token":
                            timings["ttft_ms"] = int((time.perf_counter() - started) * 1000)
                        yield event
                    chunks_retrieved = len(cached["sources"])
                    return

                decision = await self._route(opts, history)
                route = decision.route
                span.set_attribute("agent.route", route)
                yield AgentEvent("meta", {
                    "conversation_id": str(opts.conversation_id), "route": route,
                    "standalone_query": decision.standalone_query, "request_id": request_id_var.get(),
                    "cache_hit": False, "router_fallback": decision.fallback,
                })  # fmt: skip

                memories = await self._memories(opts, decision)
                candidates: list[Source] = []
                if route == "retrieve":
                    result = await self.retriever.search(decision.standalone_query, filters=opts.filters)
                    timings["retrieval_ms"] = result.retrieval_ms
                    ranked, timings["rerank_ms"] = await self.reranker.rerank(
                        decision.standalone_query, result.chunks, self.settings.final_k
                    )
                    chunks_retrieved = len(result.chunks)
                    candidates = [
                        Source(n=i, kind="chunk", label=c.location(), content=c.content, chunk_id=c.chunk_id,
                               score=c.score, source_uri=c.source_uri, title=c.title)  # fmt: skip
                        for i, c in enumerate(ranked, start=1)
                    ]
                elif route == "tool" and self.tools is not None:
                    hint = "; ".join(f"{m.subject} {m.predicate}: {m.object}" for m in memories)
                    observations = await run_tool_loop(
                        self.llm, self.tools, decision.standalone_query,
                        max_calls=self.settings.mcp_max_tool_calls, memory_hint=hint,
                    )  # fmt: skip
                    candidates = [
                        Source(n=i, kind="tool", label=f"{o.tool}({json.dumps(o.arguments)})"
                               + (" [error]" if o.is_error else ""), content=o.output)  # fmt: skip
                        for i, o in enumerate(observations, start=1)
                    ]

                built = build_prompt(
                    route=route,
                    question=decision.standalone_query,
                    candidates=candidates,
                    memories=memories,
                    history=history,
                    count=self._count,
                    context_tokens=self.settings.llm_context_tokens,
                    max_answer_tokens=self.settings.llm_max_tokens,
                )
                yield AgentEvent("sources", {
                    "sources": [_source_payload(s) for s in built.sources],
                    "memories": [{"id": m.id, "text": f"{m.subject} {m.predicate}: {m.object}",
                                  "score": round(m.score, 3)} for m in memories],
                    "dropped_sources": built.dropped_sources,
                })  # fmt: skip

                parts: list[str] = []
                async for text in self.llm.stream(built.messages, stats):
                    if not parts:
                        timings["ttft_ms"] = int((time.perf_counter() - started) * 1000)
                    parts.append(text)
                    yield AgentEvent("token", {"text": text})
                timings["llm_ms"] = stats.total_ms
                answer = "".join(parts).strip()

                parsed = extract_citations(answer, len(built.sources))
                user_id, assistant_id = await self._persist(opts, answer, built.sources, parsed.spans)
                await self.sessions.append(opts.conversation_id, opts.query, answer)
                if cacheable and route == "retrieve" and answer:
                    await self.cache.set(opts.query, {
                        "answer": answer, "standalone_query": decision.standalone_query,
                        "sources": [_source_payload(s) for s in built.sources],
                    })  # fmt: skip
                await self._enqueue_memory(user_id, assistant_id)
                yield AgentEvent("done", {
                    "message_id": assistant_id,
                    "conversation_id": str(opts.conversation_id),
                    "route": route,
                    "cache_hit": False,
                    "citations": [{"n": s.source, "chunk_id": built.sources[s.source - 1].chunk_id,
                                   "span_start": s.span_start, "span_end": s.span_end} for s in parsed.spans],
                    "invalid_citations": parsed.invalid,
                    "timings": {**timings, "total_ms": int((time.perf_counter() - started) * 1000)},
                    "usage": {"prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens},
                })  # fmt: skip
        except Exception as exc:
            log.exception("query failed")
            yield AgentEvent("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            total_s = time.perf_counter() - started
            QUERY_LATENCY.labels(route).observe(total_s)
            if timings["ttft_ms"] is not None:
                TTFT.labels(route).observe(timings["ttft_ms"] / 1000)
            if stats.tokens_per_second:
                TOKENS_PER_SECOND.observe(stats.tokens_per_second)
            await self._log_query(opts, route, timings, int(total_s * 1000), stats, chunks_retrieved, cache_hit)

    # ── steps ────────────────────────────────────────────────────
    async def _route(self, opts: QueryOptions, history: list) -> RouteDecision:
        tool_lines = self.tool_lines()
        if opts.mode == "auto":
            return await route_query(self.llm, opts.query, history, tool_lines=tool_lines)
        if opts.mode == "tool" and not tool_lines:
            raise ValueError("mode 'tool' requested but no MCP tools are available")
        if not history:
            return RouteDecision(opts.mode, opts.query)
        # Forced route, but a follow-up still needs rewriting into a standalone query.
        decision = await route_query(self.llm, opts.query, history, tool_lines=tool_lines)
        return RouteDecision(opts.mode, decision.standalone_query, decision.fallback)

    async def _memories(self, opts: QueryOptions, decision: RouteDecision) -> list:
        if not (self.settings.memory_enabled and opts.use_memory):
            return []
        try:
            vector = await self.embedder.embed_query(decision.standalone_query)
            async with self._sessionmaker() as session:
                return await retrieve_memories(session, vector, self.settings)
        except Exception as exc:  # memory is an enhancement; never fail a query on it
            log.warning("memory retrieval failed: %s", exc)
            return []

    async def _replay_cached(self, opts: QueryOptions, cached: dict, started: float) -> AsyncIterator[AgentEvent]:
        sources = [
            Source(n=s["n"], kind=s["kind"], label=s["label"], content=s.get("snippet", ""),
                   chunk_id=s.get("chunk_id"), score=s.get("score"), source_uri=s.get("source_uri"),
                   title=s.get("title"))  # fmt: skip
            for s in cached["sources"]
        ]
        answer = cached["answer"]
        yield AgentEvent("meta", {
            "conversation_id": str(opts.conversation_id), "route": "retrieve",
            "standalone_query": cached.get("standalone_query", opts.query),
            "request_id": request_id_var.get(), "cache_hit": True, "router_fallback": False,
        })  # fmt: skip
        yield AgentEvent("sources", {"sources": cached["sources"], "memories": [], "dropped_sources": 0})
        yield AgentEvent("token", {"text": answer})
        parsed = extract_citations(answer, len(sources))
        _, assistant_id = await self._persist(opts, answer, sources, parsed.spans)
        await self.sessions.append(opts.conversation_id, opts.query, answer)
        yield AgentEvent("done", {
            "message_id": assistant_id, "conversation_id": str(opts.conversation_id), "route": "retrieve",
            "cache_hit": True,
            "citations": [{"n": s.source, "chunk_id": sources[s.source - 1].chunk_id,
                           "span_start": s.span_start, "span_end": s.span_end} for s in parsed.spans],
            "invalid_citations": parsed.invalid,
            "timings": {"total_ms": int((time.perf_counter() - started) * 1000)},
            "usage": {},
        })  # fmt: skip

    async def _persist(self, opts: QueryOptions, answer: str, sources: list[Source], spans) -> tuple[int, int]:
        """User turn, assistant turn and span citations in one transaction."""
        async with self._sessionmaker() as session, session.begin():
            user_id = await session.scalar(
                insert(Message).values(conversation_id=opts.conversation_id, role="user", content=opts.query)
                .returning(Message.id)
            )
            assistant_id = await session.scalar(
                insert(Message).values(conversation_id=opts.conversation_id, role="assistant", content=answer)
                .returning(Message.id)
            )
            rows = []
            for span in spans:
                source = sources[span.source - 1]
                if source.kind == "chunk" and source.chunk_id is not None:
                    rows.append({
                        "message_id": assistant_id, "chunk_id": source.chunk_id, "rank": source.n,
                        "score": source.score, "span_start": span.span_start, "span_end": span.span_end,
                    })  # fmt: skip
            if rows:
                await session.execute(insert(Citation), rows)
            await session.execute(
                update(Conversation)
                .where(Conversation.id == opts.conversation_id, Conversation.title.is_(None))
                .values(title=opts.query[:80])
            )
        return user_id, assistant_id

    async def _enqueue_memory(self, user_id: int, assistant_id: int) -> None:
        if not (self.settings.memory_enabled and self._publish):
            return
        try:
            await self._publish(MEMORY_EXTRACT, {"user_message_id": user_id, "assistant_message_id": assistant_id})
        except Exception as exc:
            log.warning("could not enqueue memory extraction: %s", exc)

    async def _log_query(self, opts: QueryOptions, route: str, timings: dict, total_ms: int,
                         stats: StreamStats, chunks_retrieved: int, cache_hit: bool) -> None:  # fmt: skip
        try:
            async with self._sessionmaker() as session, session.begin():
                await session.execute(
                    insert(QueryLog).values(
                        conversation_id=opts.conversation_id, query=opts.query, route=route,
                        retrieval_ms=timings["retrieval_ms"], rerank_ms=timings["rerank_ms"],
                        llm_ms=timings["llm_ms"], ttft_ms=timings["ttft_ms"], total_ms=total_ms,
                        prompt_tokens=stats.prompt_tokens, completion_tokens=stats.completion_tokens,
                        model=self.llm.model, chunks_retrieved=chunks_retrieved, cache_hit=cache_hit,
                    )  # fmt: skip
                )
        except Exception as exc:
            log.warning("could not write query log: %s", exc)
