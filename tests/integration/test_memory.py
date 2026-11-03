import asyncio

import pytest
from sqlalchemy import select

from pka.core.config import get_settings
from pka.db.models import Memory
from pka.memory.resolve import (
    CandidateMemory,
    memory_history,
    memory_text,
    retract_memory,
    store_memory,
)
from pka.memory.retrieve import retrieve_memories

pytestmark = pytest.mark.integration


async def _store(sessionmaker, embedder, obj, predicate="prefers_queue_framework", confidence=0.8):
    candidate = CandidateMemory("preference", "user", predicate, obj, confidence)
    vector = (await embedder.embed([memory_text("user", predicate, obj)]))[0]
    async with sessionmaker() as s, s.begin():
        return await store_memory(s, candidate, vector, None)


async def test_conflict_supersedes_and_keeps_history(sessionmaker, embedder):
    first = await _store(sessionmaker, embedder, "Celery")
    second = await _store(sessionmaker, embedder, "arq over Celery for async codebases")
    assert first.action == "inserted"
    assert second.action == "superseded" and second.superseded_ids == [first.memory_id]

    async with sessionmaker() as s:
        old = await s.get(Memory, first.memory_id)
        new = await s.get(Memory, second.memory_id)
        assert old.superseded_by == new.id and old.valid_until is not None
        assert new.superseded_by is None and new.valid_until is None
        history = await memory_history(s, new.id)
    assert [m.object for m in history] == ["Celery", "arq over Celery for async codebases"]


async def test_restatement_reinforces_instead_of_duplicating(sessionmaker, embedder):
    a = await _store(sessionmaker, embedder, "Vim", "prefers_editor", 0.6)
    b = await _store(sessionmaker, embedder, "vim.", "prefers_editor", 0.9)
    assert b.action == "reinforced" and b.memory_id == a.memory_id
    async with sessionmaker() as s:
        rows = (await s.scalars(select(Memory))).all()
    assert len(rows) == 1 and rows[0].confidence == pytest.approx(0.9)


async def test_concurrent_writers_leave_exactly_one_active_value(sessionmaker, embedder):
    await asyncio.gather(*(_store(sessionmaker, embedder, f"value {i}") for i in range(6)))
    async with sessionmaker() as s:
        active = (
            await s.scalars(select(Memory).where(Memory.superseded_by.is_(None), Memory.valid_until.is_(None)))
        ).all()
        total = (await s.scalars(select(Memory))).all()
    assert len(active) == 1 and len(total) == 6


async def test_retrieval_uses_only_active_memories_and_counts_access(sessionmaker, embedder):
    settings = get_settings().model_copy(update={"memory_min_similarity": 0.0})
    old = await _store(sessionmaker, embedder, "Celery")
    new = await _store(sessionmaker, embedder, "arq")
    query = await embedder.embed_query("user prefers queue framework")
    async with sessionmaker() as s:
        results = await retrieve_memories(s, query, settings)
    assert [r.id for r in results] == [new.memory_id]
    async with sessionmaker() as s:
        assert (await s.get(Memory, new.memory_id)).access_count == 1
        assert (await s.get(Memory, old.memory_id)).access_count == 0


async def test_retract_ends_validity_without_deleting(sessionmaker, embedder):
    m = await _store(sessionmaker, embedder, "Emacs", "prefers_editor")
    async with sessionmaker() as s, s.begin():
        assert await retract_memory(s, m.memory_id)
    async with sessionmaker() as s, s.begin():
        assert not await retract_memory(s, m.memory_id)  # already inactive
    async with sessionmaker() as s:
        row = await s.get(Memory, m.memory_id)
    assert row is not None and row.valid_until is not None
