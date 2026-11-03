"""Test configuration. Environment is fixed *before* pka modules read settings.

Integration tests use isolated resources on the compose stack:
  Postgres database `pka_test`, Redis db 15, RabbitMQ vhost `pka_test`
so a running dev worker never consumes test messages.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

os.environ.update(
    {
        "DATABASE_URL": "postgresql+asyncpg://pka:pka@localhost:5433/pka_test",
        "READONLY_DATABASE_URL": "postgresql://pka:pka@localhost:5433/pka_test",
        "REDIS_URL": "redis://localhost:6379/15",
        "AMQP_URL": "amqp://pka:pka@localhost:5672/pka_test",
        "EMBED_MODEL": "hash",
        "EMBED_DIM": "1024",
        "RERANK_ENABLED": "false",
        "MCP_ENABLED": "false",
        "API_KEY": "test-key",
        "LOG_JSON": "false",
        "LOG_LEVEL": "WARNING",
        "QUEUE_RETRY_DELAY_MS": "200",
        "WORKER_METRICS_PORT": "0",
        "RATE_LIMIT_PER_MINUTE": "1000",
    }
)


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


INFRA_UP = all(_port_open("localhost", p) for p in (5433, 6379, 5672))


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="infra not running: docker compose up -d postgres redis rabbitmq")
    for item in items:
        if "integration" in item.keywords and not INFRA_UP:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_db():
    """Fresh schema in pka_test for the whole session."""
    url = os.environ["DATABASE_URL"]
    alembic = [sys.executable, "-m", "alembic", "-x", f"dburl={url}"]
    subprocess.run([*alembic, "downgrade", "base"], cwd=ROOT, check=True, capture_output=True)
    subprocess.run([*alembic, "upgrade", "head"], cwd=ROOT, check=True, capture_output=True)
    return url


@pytest.fixture(scope="session")
def rabbit_vhost():
    import httpx

    auth = ("pka", "pka")
    base = "http://localhost:15672/api"
    httpx.delete(f"{base}/vhosts/pka_test", auth=auth)
    httpx.put(f"{base}/vhosts/pka_test", auth=auth).raise_for_status()
    httpx.put(
        f"{base}/permissions/pka_test/pka", auth=auth, json={"configure": ".*", "write": ".*", "read": ".*"}
    ).raise_for_status()
    return os.environ["AMQP_URL"]
