"""Read-only git MCP server: log, show, diff, blame — for temporal questions static chunks
can't answer ("what did I change in this repo last week?"). Only read subcommands are
reachable, arguments are passed as a list (no shell), and repos must be inside the roots."""

import re
import subprocess

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from pka.mcp.servers._common import resolve_inside

server = MCPServer("git", instructions="Read-only git history for repositories under the corpus root.", log_level="WARNING")

MAX_OUTPUT = 20_000
_REV = re.compile(r"^[A-Za-z0-9_.\-/~^@{}:]+$")


def _git(repo: str, *args: str) -> str:
    path = resolve_inside(repo)
    if not (path / ".git").exists():
        return f"not a git repository: {repo}"
    try:
        proc = subprocess.run(
            # safe.directory: mounted repos are owned by another uid inside containers.
            ["git", "-c", "safe.directory=*", "-C", str(path), "--no-pager", *args],
            capture_output=True,
            text=True,
            timeout=15,
            env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"},
        )
    except subprocess.TimeoutExpired:
        return "git command timed out"
    out = proc.stdout if proc.returncode == 0 else f"git error: {proc.stderr.strip()}"
    return out[:MAX_OUTPUT] + ("\n… [truncated]" if len(out) > MAX_OUTPUT else "") if out else "(no output)"


def _check_rev(rev: str) -> None:
    if not _REV.match(rev) or rev.startswith("-"):
        raise ToolError(f"invalid revision: {rev!r}")


@server.tool()
def git_log(repo: str = ".", since: str = "", max_count: int = 20, path: str = "") -> str:
    """Commit history. `since` accepts git dates like "1 week ago" or "2026-10-01"."""
    args = ["log", f"--max-count={max(1, min(max_count, 200))}", "--date=iso", "--stat",
            "--format=commit %h%nAuthor: %an%nDate: %ad%n%n    %s%n"]  # fmt: skip
    if since:
        if not re.match(r"^[\w\s:.\-]+$", since):
            return "invalid since value"
        args.append(f"--since={since}")
    if path:
        args += ["--", path]
    return _git(repo, *args)


@server.tool()
def git_show(repo: str = ".", rev: str = "HEAD") -> str:
    """Show one commit: message and full diff."""
    _check_rev(rev)
    return _git(repo, "show", "--date=iso", rev)


@server.tool()
def git_diff(repo: str = ".", base: str = "HEAD", target: str = "") -> str:
    """Diff between revisions; with no `target`, the working tree against `base`."""
    _check_rev(base)
    args = ["diff", base]
    if target:
        _check_rev(target)
        args.append(target)
    return _git(repo, *args)


@server.tool()
def git_blame(repo: str, path: str, start_line: int = 1, end_line: int = 0) -> str:
    """Who last changed each line of a file (optionally a line range)."""
    resolve_inside(f"{repo}/{path}")  # the file itself must be inside the roots too
    args = ["blame", "--date=short"]
    if end_line and end_line >= start_line:
        args.append(f"-L{start_line},{end_line}")
    return _git(repo, *args, "--", path)


if __name__ == "__main__":
    server.run("stdio")
