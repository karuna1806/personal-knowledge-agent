import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from pka.agent.core import AgentEvent, QueryOptions, UnknownConversation
from pka.agent.retriever import SearchFilters
from pka.api.context import AppContext
from pka.api.deps import get_ctx, rate_limit, require_api_key
from pka.api.schemas import QueryRequest, QueryResponse

router = APIRouter(dependencies=[Depends(require_api_key)])


def _sse(event: AgentEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data, default=str)}\n\n"


async def _options(ctx: AppContext, body: QueryRequest) -> QueryOptions:
    try:
        conversation_id: uuid.UUID = await ctx.agent.ensure_conversation(body.conversation_id)
    except UnknownConversation as exc:
        raise HTTPException(404, f"unknown conversation {exc}") from exc
    filters = None
    if body.filters:
        filters = SearchFilters(body.filters.source_types, body.filters.language, body.filters.source_prefix)
    return QueryOptions(
        query=body.query.strip(),
        conversation_id=conversation_id,
        mode=body.mode,
        filters=filters,
        use_memory=body.use_memory,
        use_cache=body.use_cache,
    )


@router.post(
    "/query",
    dependencies=[Depends(rate_limit)],
    response_model=QueryResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream when stream=true"}},
)
async def query(body: QueryRequest, ctx: AppContext = Depends(get_ctx)):
    """Ask a question. `stream=true` (default) returns Server-Sent Events:
    `meta` → `sources` → `token`* → `done` (or `error`). Closing the connection cancels."""
    opts = await _options(ctx, body)
    if body.stream:

        async def events():
            async for event in ctx.agent.run(opts):
                yield _sse(event)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    meta: dict = {}
    sources: dict = {}
    done: dict = {}
    parts: list[str] = []
    async for event in ctx.agent.run(opts):
        if event.type == "meta":
            meta = event.data
        elif event.type == "sources":
            sources = event.data
        elif event.type == "token":
            parts.append(event.data["text"])
        elif event.type == "done":
            done = event.data
        elif event.type == "error":
            raise HTTPException(502, event.data["message"])
    return QueryResponse(
        conversation_id=opts.conversation_id,
        message_id=done.get("message_id"),
        route=done.get("route", meta.get("route", "unknown")),
        answer="".join(parts).strip(),
        sources=sources.get("sources", []),
        citations=done.get("citations", []),
        invalid_citations=done.get("invalid_citations", []),
        memories=sources.get("memories", []),
        cache_hit=bool(done.get("cache_hit")),
        timings=done.get("timings", {}),
        request_id=meta.get("request_id"),
    )
