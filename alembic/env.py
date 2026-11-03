import asyncio
import os

from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from pka.core.config import get_settings

config = context.config


def _url() -> str:
    # `-x dburl=...` wins (used by the test suite), then DATABASE_URL via settings.
    return context.get_x_argument(as_dictionary=True).get("dburl") or os.environ.get(
        "DATABASE_URL", get_settings().database_url
    )


def run_migrations_offline() -> None:
    context.configure(url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection) -> None:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_url())
    async with engine.connect() as connection:
        await connection.run_sync(_do_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
