"""Read-only SQL over the agent's own metadata ("how many notes did I write in March?").

Three independent guards: a role with only SELECT grants (infra/postgres/init), a READ ONLY
transaction, and a statement allowlist. Any one of them alone would block writes."""

import os
import re

import asyncpg
from mcp.server.mcpserver import MCPServer

server = MCPServer("postgres", instructions="Read-only SQL over the knowledge agent's metadata tables.", log_level="WARNING")

MAX_ROWS = 100
_ALLOWED = re.compile(r"^\s*(select|with|explain|table|values)\b", re.I)
_TABLES = ("documents", "chunks", "memories", "conversations", "messages", "query_logs", "ingest_jobs")


def _dsn() -> str:
    dsn = os.environ.get("READONLY_DATABASE_URL", "postgresql://pka_ro:pka_ro@localhost:5433/pka")
    return dsn.replace("postgresql+asyncpg://", "postgresql://")


@server.tool()
async def describe_tables() -> str:
    """Columns of the queryable tables (documents, chunks, memories, messages, query_logs, ...)."""
    conn = await asyncpg.connect(_dsn(), timeout=5)
    try:
        rows = await conn.fetch(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = ANY($1::text[]) "
            "ORDER BY table_name, ordinal_position",
            list(_TABLES),
        )
    finally:
        await conn.close()
    out: dict[str, list[str]] = {}
    for r in rows:
        if r["data_type"] in ("USER-DEFINED",) and r["column_name"] in ("embedding",):
            continue  # vectors are noise for SQL questions
        out.setdefault(r["table_name"], []).append(f"{r['column_name']} {r['data_type']}")
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in out.items())


@server.tool()
async def query(sql: str) -> str:
    """Run one read-only SELECT. Returns up to 100 rows as tab-separated text.
    Example: SELECT source_type, count(*) FROM documents GROUP BY 1"""
    statement = sql.strip().rstrip(";")
    if ";" in statement:
        return "only a single statement is allowed"
    if not _ALLOWED.match(statement):
        return "only SELECT / WITH queries are allowed"
    conn = await asyncpg.connect(_dsn(), timeout=5)
    try:
        async with conn.transaction(readonly=True):
            await conn.execute("SET LOCAL statement_timeout = '5s'")
            records = await conn.fetch(statement)
    except asyncpg.PostgresError as exc:
        return f"SQL error: {exc}"
    finally:
        await conn.close()
    if not records:
        return "(0 rows)"
    columns = list(records[0].keys())
    lines = ["\t".join(columns)]
    for record in records[:MAX_ROWS]:
        values = []
        for value in record.values():
            text = str(value)
            values.append(text[:200] + "…" if len(text) > 200 else text)
        lines.append("\t".join(values))
    if len(records) > MAX_ROWS:
        lines.append(f"… ({len(records) - MAX_ROWS} more rows)")
    return "\n".join(lines)


if __name__ == "__main__":
    server.run("stdio")
