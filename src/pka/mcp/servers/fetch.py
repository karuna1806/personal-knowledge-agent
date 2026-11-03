"""Fetch a URL and return its readable text. Read-only: it does not queue the page for
ingestion — that is an explicit user action (POST /ingest {"url": ...})."""

from mcp.server.mcpserver import MCPServer

from pka.ingest.loaders.base import UnsupportedSource
from pka.ingest.loaders.web import load_url

server = MCPServer("fetch", instructions="Retrieve a web page as text.", log_level="WARNING")

MAX_CHARS = 12_000


@server.tool()
async def fetch_url(url: str) -> str:
    """Fetch an http(s) URL and return its main text content as markdown."""
    try:
        doc = await load_url(url)
    except UnsupportedSource as exc:
        return f"cannot fetch: {exc}"
    except Exception as exc:
        return f"fetch failed: {type(exc).__name__}: {exc}"
    text = doc.content
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n… [truncated]"
    return f"# {doc.title}\n\n{text}" if doc.title else text


if __name__ == "__main__":
    server.run("stdio")
