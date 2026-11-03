"""Read-only filesystem MCP server, confined to MCP_ALLOWED_ROOTS (default CORPUS_ROOT)."""

import fnmatch
import os
from datetime import UTC, datetime

from mcp.server.mcpserver import MCPServer

from pka.ingest.loaders.base import IGNORED_DIRS
from pka.mcp.servers._common import allowed_roots, resolve_inside

server = MCPServer("filesystem", instructions="Read-only access to the user's local files.", log_level="WARNING")

MAX_READ_BYTES = 200_000


@server.tool()
def list_directory(path: str = ".") -> str:
    """List files and folders in a directory (relative to the corpus root)."""
    target = resolve_inside(path)
    if not target.is_dir():
        return f"not a directory: {path}"
    lines = []
    for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:500]:
        if entry.name.startswith(".") and entry.name not in {".env.example"}:
            continue
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            stat = entry.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, UTC).strftime("%Y-%m-%d %H:%M")
            lines.append(f"{entry.name}\t{stat.st_size} bytes\tmodified {modified}")
    return "\n".join(lines) or "(empty directory)"


@server.tool()
def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
    """Read a text file (optionally a line range). Paths are relative to the corpus root."""
    target = resolve_inside(path)
    if not target.is_file():
        return f"not a file: {path}"
    raw = target.read_bytes()[:MAX_READ_BYTES]
    if b"\x00" in raw[:8192]:
        return "binary file; not shown"
    lines = raw.decode("utf-8", errors="replace").split("\n")
    start = max(1, start_line)
    selected = lines[start - 1 : start - 1 + max(1, min(max_lines, 2000))]
    return "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(selected))


@server.tool()
def search_files(pattern: str, path: str = ".", content: str = "") -> str:
    """Find files whose name matches a glob `pattern` (e.g. "*.md"), optionally only those
    containing the text `content` (case-insensitive). Returns up to 100 paths."""
    base = resolve_inside(path)
    root = allowed_roots()[0]
    needle = content.lower()
    hits = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]
        for name in filenames:
            if not fnmatch.fnmatch(name, pattern):
                continue
            full = os.path.join(dirpath, name)
            if needle:
                try:
                    with open(full, "rb") as handle:
                        if needle not in handle.read(MAX_READ_BYTES).decode("utf-8", "ignore").lower():
                            continue
                except OSError:
                    continue
            try:
                hits.append(os.path.relpath(full, root))
            except ValueError:
                hits.append(full)
            if len(hits) >= 100:
                return "\n".join(hits) + "\n… (more results not shown)"
    return "\n".join(hits) or "no matches"


if __name__ == "__main__":
    server.run("stdio")
