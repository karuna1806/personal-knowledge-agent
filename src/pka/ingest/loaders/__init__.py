"""Loaders turn a file path or URL into a LoadedDocument. They do no chunking."""

from pka.ingest.loaders.base import (
    IGNORED_DIRS,
    LoadedDocument,
    UnsupportedSource,
    classify_path,
    iter_corpus_files,
    load_path,
)
from pka.ingest.loaders.web import load_url

__all__ = [
    "IGNORED_DIRS",
    "LoadedDocument",
    "UnsupportedSource",
    "classify_path",
    "iter_corpus_files",
    "load_path",
    "load_url",
]
