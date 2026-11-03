"""Memory extraction — runs in a worker off `memory.extract`, never in the request path:
it's a second LLM call the user doesn't see, so inline it would only add latency (§5)."""

import logging

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pka.core.versions import MEMORY_VERSION_KEY, bump
from pka.db.models import Message
from pka.ingest.embed import Embedder
from pka.llm.client import LLMClient
from pka.memory.resolve import (
    KINDS,
    CandidateMemory,
    StoreResult,
    memory_text,
    store_memory,
    validate,
)
from pka.memory.retrieve import nearest_active

log = logging.getLogger(__name__)
KNOWN_CONTEXT = 10

MIN_CONFIDENCE = 0.3
MAX_PER_TURN = 5

SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "subject", "predicate", "object", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}

SYSTEM = """You extract durable memories about the USER from one conversation turn.

Extract only facts the USER states or clearly implies about themselves: their preferences,
their projects, tools and technologies they use, people/organisations they work with, and
stable facts about their situation. Ignore:
- anything that only appears in the ASSISTANT's reply or in quoted documents,
- questions (asking about X is not a fact about the user),
- transient details (today's mood, a one-off task).

Each memory is a triple:
- kind: preference | fact | project | entity
- subject: "user" for the user; otherwise the project/entity name exactly as written
- predicate: short snake_case relation, reusing a general form so later values can replace
  earlier ones, e.g. prefers_language, prefers_queue_framework, works_on, works_at, uses_editor,
  lives_in, project_uses_database
- object: the value, concise
- confidence: 0.0-1.0 — how explicitly the user stated it (explicit statement ≈ 0.9,
  strong implication ≈ 0.6)

Return {"memories": []} when the turn contains nothing durable. That is the common case."""


def build_messages(user_text: str, assistant_text: str, known: list[str] | None = None) -> list[dict]:
    """`known`: related memories already stored, as "subject predicate: object" lines. Shown so
    an update reuses the existing key — without it a small model writes `prefers_editor`
    one day and `uses_editor` the next, and supersession never fires (both stay 'true')."""
    known_block = ""
    if known:
        known_block = (
            "ALREADY STORED about the user (most related first). If this turn updates or "
            "contradicts one of these, output the SAME subject and predicate with the new object, "
            "so the new value replaces the old one. Don't re-output unchanged ones.\n"
            + "\n".join(f"- {line}" for line in known)
            + "\n\n"
        )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": known_block
            + f"USER said:\n<<<\n{user_text[:4000]}\n>>>\n\n"
            f"ASSISTANT replied (context only — do not extract from this):\n<<<\n{assistant_text[:1500]}\n>>>",
        },
    ]


def parse_candidates(raw: dict) -> list[CandidateMemory]:
    out: list[CandidateMemory] = []
    for item in (raw.get("memories") or [])[:MAX_PER_TURN]:
        if not isinstance(item, dict):
            continue
        try:
            candidate = CandidateMemory(
                kind=str(item.get("kind", "")),
                subject=str(item.get("subject", "")),
                predicate=str(item.get("predicate", "")),
                object=str(item.get("object", "")),
                confidence=float(item.get("confidence", 0.5)),
            )
        except (TypeError, ValueError):
            continue
        valid = validate(candidate)
        if valid and valid.confidence >= MIN_CONFIDENCE:
            out.append(valid)
    return out


async def extract_and_store(
    sessionmaker: async_sessionmaker[AsyncSession],
    llm: LLMClient,
    embedder: Embedder,
    *,
    user_message_id: int,
    assistant_message_id: int,
    redis: Redis | None = None,
) -> list[StoreResult]:
    async with sessionmaker() as session:
        user_msg = await session.get(Message, user_message_id)
        assistant_msg = await session.get(Message, assistant_message_id)
    if user_msg is None:
        log.info("source message gone; nothing to extract", extra={"message_id": user_message_id})
        return []

    query_vector = await embedder.embed_query(user_msg.content[:2000])
    async with sessionmaker() as session:
        related = await nearest_active(session, query_vector, KNOWN_CONTEXT)
    known = [f"{m.subject} {m.predicate}: {m.object}" for m in related]

    raw = await llm.complete_json(
        build_messages(user_msg.content, assistant_msg.content if assistant_msg else "", known),
        SCHEMA,
        name="memories",
        max_tokens=600,
    )
    candidates = parse_candidates(raw)
    if not candidates:
        return []
    vectors = await embedder.embed([memory_text(c.subject, c.predicate, c.object) for c in candidates])
    results: list[StoreResult] = []
    for candidate, vector in zip(candidates, vectors, strict=True):
        async with sessionmaker() as session, session.begin():
            result = await store_memory(session, candidate, vector, user_message_id)
        results.append(result)
        log.info(
            "memory %s", result.action,
            extra={"memory_id": result.memory_id, "subject": candidate.subject,
                   "predicate": candidate.predicate, "superseded": result.superseded_ids},
        )  # fmt: skip
    if any(r.action in ("inserted", "superseded") for r in results):
        await bump(redis, MEMORY_VERSION_KEY)  # cached answers may depend on what it now believes
    return results
