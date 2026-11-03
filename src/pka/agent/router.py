"""Router-first, not agent-first (§3.7): one constrained classification picks exactly one
path. A single decision validated against a fixed enum is far more reliable for a 7-12B
model than an open ReAct loop.

The same call rewrites follow-ups into a standalone query ("what about its DLQ?" →
"What is RabbitMQ's dead-letter queue used for?") so retrieval sees the real question."""

import logging
from dataclasses import dataclass

from pka.llm.client import LLMClient, Message

log = logging.getLogger(__name__)

ROUTES = ("retrieve", "tool", "direct")


@dataclass
class RouteDecision:
    route: str
    standalone_query: str
    fallback: bool = False


def route_schema(routes: list[str], *, rewrite: bool) -> dict:
    properties: dict = {"route": {"type": "string", "enum": routes}}
    if rewrite:
        properties["standalone_query"] = {"type": "string"}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_router_messages(query: str, history: list[Message], tool_lines: list[str]) -> list[Message]:
    routes = [
        "- retrieve: the answer would be in the user's own notes, documents or code (their "
        "designs, decisions, explanations, how something of theirs works). DEFAULT — when in "
        "doubt, choose retrieve.",
    ]
    if tool_lines:
        routes.append(
            "- tool: needs LIVE data the notes can't hold: current files on disk, git history "
            "or recent changes, counts/statistics over the user's database, or fetching a URL. "
            "Available tools:\n" + "\n".join(f"    {line}" for line in tool_lines)
        )
    routes.append(
        "- direct: greetings, thanks, the user TELLING you something about themselves (a "
        "preference, a fact, an update — not a question), or requests about this conversation "
        "itself (summarise/rephrase what was said). Never for factual questions."
    )
    system = "You route questions for a personal knowledge assistant. Pick one route:\n" + "\n".join(routes)
    if history:
        system += (
            "\n\nAlso write standalone_query: the user's latest message rewritten to be "
            "self-contained, resolving pronouns and references from the conversation. If it is "
            "already self-contained, copy it unchanged."
        )
    transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:500]}" for m in history[-6:])
    user = (f"Conversation so far:\n{transcript}\n\n" if transcript else "") + f"Latest message: {query}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def route_query(
    llm: LLMClient, query: str, history: list[Message], *, tool_lines: list[str]
) -> RouteDecision:
    routes = [r for r in ROUTES if r != "tool" or tool_lines]
    # With no history there is nothing to resolve, so don't spend ~1s generating a copy of
    # the query: emit the route alone.
    rewrite = bool(history)
    try:
        raw = await llm.complete_json(
            build_router_messages(query, history, tool_lines),
            route_schema(routes, rewrite=rewrite),
            name="route",
            max_tokens=200 if rewrite else 20,
        )
        route = raw.get("route")
        standalone = (str(raw.get("standalone_query") or "").strip() if rewrite else "") or query
        if route not in routes:
            raise ValueError(f"route {route!r} not in {routes}")
        return RouteDecision(route, standalone)
    except Exception as exc:
        # Degrade to the safe default rather than failing the request.
        log.warning("router failed, defaulting to retrieve: %s", exc)
        return RouteDecision("retrieve", query, fallback=True)
