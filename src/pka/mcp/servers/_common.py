import os
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError


class AccessDenied(ToolError):
    """A ToolError's message reaches the model (other exceptions are masked as crashes)."""


def allowed_roots() -> list[Path]:
    raw = os.environ.get("MCP_ALLOWED_ROOTS") or os.environ.get("CORPUS_ROOT") or "."
    return [Path(p).expanduser().resolve() for p in raw.split(os.pathsep) if p]


def resolve_inside(path: str, roots: list[Path] | None = None) -> Path:
    """Resolve `path` (absolute, or relative to the first root) and refuse anything outside
    the allowed roots — including via symlinks and `..`."""
    roots = roots or allowed_roots()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    resolved = candidate.resolve()
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    raise AccessDenied(f"{path} is outside the allowed roots ({', '.join(map(str, roots))})")
