"""MCP server configuration and the tool allowlist. Every server shipped here is read-only
(§3.7 rule 4): tool results are context, never actions."""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_VAR = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


@dataclass
class ServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] | None = None  # None = all tools the server exposes
    enabled: bool = True


def _expand(value: str) -> str:
    def sub(match: re.Match) -> str:
        if match.group(1) == "PYTHON":
            return sys.executable
        return os.environ.get(match.group(1), match.group(2) or "")

    return _VAR.sub(sub, value)


def load_registry(path: Path) -> list[ServerConfig]:
    if not Path(path).is_file():
        return []
    raw = yaml.safe_load(Path(path).read_text()) or {}
    servers = []
    for name, spec in (raw.get("servers") or {}).items():
        if not isinstance(spec, dict):
            continue
        enabled = spec.get("enabled", True)
        if isinstance(enabled, str):
            enabled = _expand(enabled).strip().lower() not in {"0", "false", "no", "off", ""}
        servers.append(
            ServerConfig(
                name=name,
                command=_expand(str(spec["command"])),
                args=[_expand(str(a)) for a in spec.get("args", [])],
                env={k: _expand(str(v)) for k, v in (spec.get("env") or {}).items()},
                allowed_tools=spec.get("allowed_tools"),
                enabled=bool(enabled),
            )
        )
    return servers
