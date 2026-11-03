"""The whole request path through the real FastAPI app (lifespan included), real Postgres,
Redis and RabbitMQ, a real in-process worker, and the stub OpenAI-compatible LLM."""

import asyncio
import json
import os

import httpx
import pytest

from tests import fake_llm

pytestmark = pytest.mark.integration
KEY = {"x-api-key": "test-key"}

NOTES = {
    "notes/queue.md": "# Queue design\n\n## Why RabbitMQ\n\nDead-letter exchanges keep failed ingests inspectable.\n",
    "notes/index.md": "# Index\n\n## HNSW\n\nHNSW needs no periodic rebuilds, unlike IVFFlat.\n",
    "code/broker.py": "def connect_broker(url):\n    return url\n",
}


@pytest.fixture(scope="module")
def llm_url():
    url, server = fake_llm.start()
    yield url
    server.should_exit = True


@pytest.fixture
async def app_client(llm_url, tmp_path, migrated_db, rabbit_vhost, redis):
    from pka.core.config import get_settings

    corpus = tmp_path / "corpus"
    for rel, text in NOTES.items():
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_text(text)
    os.environ.update({"LLM_BASE_URL": llm_url, "LLM_MODEL": "fake", "CORPUS_ROOT": str(corpus),
                       "UPLOAD_DIR": str(tmp_path / "uploads")})  # fmt: skip
    get_settings.cache_clear()
    from pka.api.main import create_app
    from pka.workers.runner import Worker, build_context

    app = create_app()
    async with app.router.lifespan_context(app):
        ctx = app.state.ctx
        async with ctx.broker.connection.channel() as ch:
            from pka.queue import broker as b

            for name in [s.name for s in b.QUEUES.values()] + [b.RETRY_QUEUE, b.DLQ]:
                await (await ch.declare_queue(name, passive=True)).purge()
        worker = Worker(await build_context(get_settings()), ["ingest", "memory"])
        await worker.start()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
            client.app = app
            yield client
        await worker.stop(timeout_s=5)
        await worker.ctx.llm.close()
        await worker.ctx.redis.aclose()
    get_settings.cache_clear()


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


async def ingest_corpus(client) -> None:
    r = await client.post("/ingest", json={"path": "."}, headers=KEY)
    assert r.status_code == 202, r.text
    job_ids = [j["id"] for j in r.json()["jobs"]]
    assert len(job_ids) == 3
    for _ in range(100):
        statuses = [(await client.get(f"/ingest/jobs/{j}", headers=KEY)).json()["status"] for j in job_ids]
        if all(s == "done" for s in statuses):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"ingest did not finish: {statuses}")


async def ask(client, query, **extra) -> list[tuple[str, dict]]:
    r = await client.post("/query", json={"query": query, **extra}, headers=KEY)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return parse_sse(r.text)


async def test_health_ready_auth_and_validation(app_client):
    assert (await app_client.get("/health")).json() == {"status": "ok"}
    ready = await app_client.get("/ready")
    assert ready.status_code == 200 and all(ready.json()["checks"].values())
    assert (await app_client.post("/query", json={"query": "x"})).status_code == 401
    assert (await app_client.post("/query", json={"query": "x"}, headers={"x-api-key": "wrong"})).status_code == 401
    assert (await app_client.post("/query", json={"query": ""}, headers=KEY)).status_code == 422
    assert (await app_client.post("/ingest", json={"path": "../../etc"}, headers=KEY)).status_code == 400
    assert (await app_client.post("/ingest", json={"path": "missing.md"}, headers=KEY)).status_code == 404
    assert (await app_client.post("/ingest", json={}, headers=KEY)).status_code == 422
    assert "x-request-id" in (await app_client.get("/health")).headers
    metrics = (await app_client.get("/metrics")).text
    assert "pka_queue_depth" in metrics and "pka_dlq_depth" in metrics


async def test_query_streams_grounded_answer_with_persisted_citations(app_client):
    await ingest_corpus(app_client)
    events = await ask(app_client, "Why RabbitMQ dead-letter exchanges?")
    types = [t for t, _ in events]
    assert types[0] == "meta" and types[1] == "sources" and types[-1] == "done"
    assert all(t == "token" for t in types[2:-1])
    meta, sources, done = events[0][1], events[1][1], events[-1][1]
    assert meta["route"] == "retrieve" and not meta["cache_hit"]
    assert sources["sources"][0]["source_uri"] == "notes/queue.md"
    answer = "".join(d["text"] for t, d in events if t == "token")
    assert answer.endswith("[1].")
    assert done["citations"] and done["citations"][0]["chunk_id"] == sources["sources"][0]["chunk_id"]
    span = done["citations"][0]
    assert answer.strip()[span["span_start"] : span["span_end"]].startswith("From the notes")

    messages = (await app_client.get(f"/conversations/{meta['conversation_id']}/messages", headers=KEY)).json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["citations"][0]["source_uri"] == "notes/queue.md"

    chunk = (await app_client.get(f"/chunks/{span['chunk_id']}", headers=KEY)).json()
    assert "Dead-letter" in chunk["content"]


async def test_query_cache_and_invalidation(app_client):
    await ingest_corpus(app_client)
    first = await ask(app_client, "Why RabbitMQ dead-letter exchanges?")
    second = await ask(app_client, "  why rabbitmq DEAD-LETTER exchanges ")
    assert second[0][1]["cache_hit"] is True and second[-1][1]["cache_hit"] is True
    assert "".join(d["text"] for t, d in first if t == "token").strip() == second[2][1]["text"]
    # Deleting a document bumps the corpus version → cache miss
    doc_id = (await app_client.get("/documents?q=index", headers=KEY)).json()[0]["id"]
    assert (await app_client.delete(f"/documents/{doc_id}", headers=KEY)).status_code == 204
    third = await ask(app_client, "Why RabbitMQ dead-letter exchanges?")
    assert third[0][1]["cache_hit"] is False


async def test_follow_up_is_rewritten_and_not_cached(app_client):
    await ingest_corpus(app_client)
    first = await ask(app_client, "Why RabbitMQ dead-letter exchanges?")
    conversation_id = first[0][1]["conversation_id"]
    follow = await ask(app_client, "and what about HNSW?", conversation_id=conversation_id)
    assert follow[0][1]["standalone_query"] == "and what about HNSW? (standalone)"
    assert follow[0][1]["cache_hit"] is False
    unknown = await app_client.post("/query", headers=KEY, json={
        "query": "x", "conversation_id": "00000000-0000-0000-0000-000000000000"})  # fmt: skip
    assert unknown.status_code == 404


async def test_direct_route_and_json_mode(app_client):
    r = await app_client.post("/query", json={"query": "hello there", "stream": False}, headers=KEY)
    body = r.json()
    assert r.status_code == 200 and body["route"] == "direct" and body["sources"] == []
    assert body["answer"] == "I don't know based on the available sources."


async def test_memory_extraction_and_supersession_via_worker(app_client):
    first = await ask(app_client, "I prefer Celery for task queues", mode="direct")
    conversation_id = first[0][1]["conversation_id"]

    async def active():
        return (await app_client.get("/memories", headers=KEY)).json()

    for _ in range(150):
        if mems := await active():
            break
        await asyncio.sleep(0.1)
    assert [(m["predicate"], m["object"]) for m in mems] == [("prefers_tool", "Celery")]

    await ask(app_client, "Actually I prefer arq now", mode="direct", conversation_id=conversation_id)
    for _ in range(150):
        mems = await active()
        if mems and mems[0]["object"] == "arq":
            break
        await asyncio.sleep(0.1)
    assert [(m["object"], m["active"]) for m in mems] == [("arq", True)]
    history = (await app_client.get(f"/memories/{mems[0]['id']}/history", headers=KEY)).json()
    assert [(m["object"], m["active"]) for m in history] == [("Celery", False), ("arq", True)]
    assert history[0]["superseded_by"] == mems[0]["id"]

    # A cached answer must not outlive a change in what the agent believes.
    await ingest_corpus(app_client)
    assert (await ask(app_client, "Why RabbitMQ dead-letter exchanges?"))[0][1]["cache_hit"] is False
    assert (await ask(app_client, "Why RabbitMQ dead-letter exchanges?"))[0][1]["cache_hit"] is True
    retracted = (await app_client.post(f"/memories/{mems[0]['id']}/retract", headers=KEY)).json()
    assert retracted["active"] is False
    assert await active() == []
    assert (await ask(app_client, "Why RabbitMQ dead-letter exchanges?"))[0][1]["cache_hit"] is False


async def test_upload_and_rate_limit(app_client):
    files = {"file": ("design.md", b"# Upload\n\nUploaded body.\n", "text/markdown")}
    r = await app_client.post("/ingest/upload", files=files, headers=KEY)
    assert r.status_code == 202 and r.json()["jobs"][0]["target"] == "upload/design.md"
    bad = await app_client.post("/ingest/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, headers=KEY)
    assert bad.status_code == 415

    app_client.app.state.ctx.settings.rate_limit_per_minute = 2
    try:
        codes = [(await app_client.post("/ingest", json={"url": "ftp://x"}, headers=KEY)).status_code for _ in range(4)]
    finally:
        app_client.app.state.ctx.settings.rate_limit_per_minute = 1000
    assert codes[-1] == 429
