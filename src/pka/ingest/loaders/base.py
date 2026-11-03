from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

NOTE_EXTENSIONS = {".md", ".markdown", ".mdx", ".txt", ".rst", ".org"}
PDF_EXTENSIONS = {".pdf"}
# Grammars bundled in the Docker image (see Dockerfile). Others are downloaded on first use.
CODE_LANGUAGES = {
    "python", "javascript", "typescript", "tsx", "go", "rust", "java", "c", "cpp", "csharp",
    "ruby", "php", "kotlin", "swift", "scala", "lua", "bash", "shell", "sql", "html", "css",
    "yaml", "toml", "json",
}  # fmt: skip
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next", ".idea", ".vscode", "target",
}  # fmt: skip
MAX_FILE_BYTES = 5 * 1024 * 1024


class UnsupportedSource(ValueError):
    pass


@dataclass
class LoadedDocument:
    source_type: str  # note | code | pdf | web
    source_uri: str
    title: str | None
    content: str
    metadata: dict = field(default_factory=dict)
    pages: list[str] | None = None  # PDFs: text per page, page n = pages[n-1]


def detect_code_language(path: Path) -> str | None:
    try:
        from tree_sitter_language_pack import detect_language_from_path
    except ImportError:
        return None
    lang = detect_language_from_path(str(path))
    return lang if lang in CODE_LANGUAGES else None


def classify_path(path: Path) -> tuple[str, str | None]:
    """Return (source_type, language). Raises UnsupportedSource for anything else."""
    suffix = path.suffix.lower()
    if suffix in NOTE_EXTENSIONS:
        return "note", None
    if suffix in PDF_EXTENSIONS:
        return "pdf", None
    language = detect_code_language(path)
    if language:
        return "code", language
    raise UnsupportedSource(f"unsupported file type: {path.name}")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise UnsupportedSource(f"binary file: {path.name}")
    return raw.decode("utf-8", errors="replace")


def load_path(path: Path, source_uri: str | None = None) -> LoadedDocument:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.stat().st_size > MAX_FILE_BYTES and path.suffix.lower() not in PDF_EXTENSIONS:
        raise UnsupportedSource(f"file too large: {path.name}")
    source_type, language = classify_path(path)
    uri = source_uri or str(path)

    if source_type == "pdf":
        from pka.ingest.loaders.pdf import load_pdf_bytes

        return load_pdf_bytes(path.read_bytes(), uri, fallback_title=path.stem)

    text = _read_text(path)
    if source_type == "note":
        return LoadedDocument("note", uri, _markdown_title(text) or path.stem, text, {})
    return LoadedDocument(
        "code", uri, path.name, text, {"language": language, "path": str(path)}
    )


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines()[:50]:
        if line.startswith("# "):
            return line[2:].strip()
    return None


def iter_corpus_files(root: Path) -> Iterator[Path]:
    """Every ingestible file under `root`, skipping VCS/dependency/build directories."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            classify_path(path)
        except UnsupportedSource:
            continue
        yield path
