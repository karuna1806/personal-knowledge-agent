"""Pydantic models for every request/response boundary."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Filters(BaseModel):
    source_types: list[Literal["note", "code", "pdf", "web"]] | None = None
    language: str | None = None
    source_prefix: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None
    mode: Literal["auto", "retrieve", "tool", "direct"] = "auto"
    stream: bool = True
    filters: Filters | None = None
    use_memory: bool = True
    use_cache: bool = True


class SourceOut(BaseModel):
    n: int
    kind: str
    label: str
    chunk_id: int | None = None
    score: float | None = None
    source_uri: str | None = None
    title: str | None = None
    snippet: str


class CitationOut(BaseModel):
    n: int
    chunk_id: int | None
    span_start: int
    span_end: int


class QueryResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: int | None
    route: str
    answer: str
    sources: list[SourceOut]
    citations: list[CitationOut]
    invalid_citations: list[int]
    memories: list[dict]
    cache_hit: bool
    timings: dict
    request_id: str | None = None


class IngestRequest(BaseModel):
    path: str | None = Field(default=None, description="File or directory, relative to CORPUS_ROOT or absolute inside it")
    url: str | None = None

    @model_validator(mode="after")
    def exactly_one(self):
        if bool(self.path) == bool(self.url):
            raise ValueError("provide exactly one of 'path' or 'url'")
        return self


class JobOut(BaseModel):
    id: uuid.UUID
    kind: str
    target: str
    status: str
    attempts: int
    document_id: int | None
    chunks: int | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IngestAccepted(BaseModel):
    jobs: list[JobOut]
    skipped_files: int = 0


class DocumentOut(BaseModel):
    id: int
    source_type: str
    source_uri: str
    title: str | None
    content_hash: str
    metadata: dict
    ingested_at: datetime | None
    chunk_count: int | None = None


class ChunkOut(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    token_count: int
    embed_model: str
    metadata: dict
    source_uri: str
    title: str | None
    source_type: str


class MemoryOut(BaseModel):
    id: int
    kind: str
    subject: str
    predicate: str
    object: str
    confidence: float
    source_msg_id: int | None
    superseded_by: int | None
    valid_from: datetime
    valid_until: datetime | None
    access_count: int
    last_accessed: datetime | None
    active: bool

    model_config = {"from_attributes": True}
