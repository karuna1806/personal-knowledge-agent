"""Ingestion is always async (§2): the API validates, records a job, publishes, returns 202.
Embedding a 200-page PDF takes tens of seconds and must never sit inside an HTTP request."""

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import insert, select, update

from pka.api.context import AppContext
from pka.api.deps import get_ctx, rate_limit, require_api_key
from pka.api.schemas import IngestAccepted, IngestRequest, JobOut
from pka.db.models import IngestJob
from pka.ingest.loaders import UnsupportedSource, classify_path, iter_corpus_files
from pka.queue.broker import INGEST_FILE, INGEST_URL

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_api_key)])

MAX_FILES_PER_REQUEST = 10_000


def _resolve_corpus_path(ctx: AppContext, raw: str) -> tuple[Path, Path]:
    """Resolve a user-supplied path, refusing anything outside CORPUS_ROOT."""
    root = ctx.settings.corpus_root.expanduser().resolve()
    candidate = Path(raw).expanduser()
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(400, f"path must be inside CORPUS_ROOT ({root})")
    if not resolved.exists():
        raise HTTPException(404, f"not found: {raw}")
    return resolved, root


async def _create_jobs(ctx: AppContext, items: list[tuple[str, str, dict]]) -> list[IngestJob]:
    """items: (kind, target, payload). Jobs are recorded before publishing, so a publish
    failure leaves an explicit 'failed' job rather than a silent gap."""
    if not ctx.broker_ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "message broker unavailable")
    async with ctx.sessionmaker() as session, session.begin():
        jobs = list(
            (
                await session.scalars(
                    insert(IngestJob).returning(IngestJob),
                    [{"kind": kind, "target": target, "status": "queued"} for kind, target, _ in items],
                )
            ).all()
        )
    failed: list[uuid.UUID] = []
    for job, (kind, _, payload) in zip(jobs, items, strict=True):
        try:
            await ctx.publish(INGEST_FILE if kind == "file" else INGEST_URL, {**payload, "job_id": str(job.id)})
        except Exception as exc:
            failed.append(job.id)
            job.status, job.error = "failed", f"publish failed: {exc}"
    if failed:
        async with ctx.sessionmaker() as session, session.begin():
            await session.execute(
                update(IngestJob).where(IngestJob.id.in_(failed)).values(status="failed", error="publish failed")
            )
    return jobs


@router.post("", status_code=202, response_model=IngestAccepted, dependencies=[Depends(rate_limit)])
async def ingest(body: IngestRequest, ctx: AppContext = Depends(get_ctx)) -> IngestAccepted:
    """Queue a file, a whole directory (recursively), or a URL for ingestion."""
    if body.url:
        if not re.match(r"^https?://", body.url):
            raise HTTPException(400, "url must be http(s)")
        jobs = await _create_jobs(ctx, [("url", body.url, {"url": body.url})])
        return IngestAccepted(jobs=[JobOut.model_validate(j) for j in jobs])

    path, root = _resolve_corpus_path(ctx, body.path or "")
    if path.is_file():
        try:
            classify_path(path)
        except UnsupportedSource as exc:
            raise HTTPException(415, str(exc)) from exc
        files = [path]
    else:
        files = list(iter_corpus_files(path))
        if not files:
            raise HTTPException(400, "no ingestible files found in directory")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(413, f"{len(files)} files; split into requests of at most {MAX_FILES_PER_REQUEST}")
    items = []
    for f in files:
        source_uri = str(f.relative_to(root)) if f != root else f.name
        items.append(("file", source_uri, {"path": str(f), "source_uri": source_uri}))
    jobs = await _create_jobs(ctx, items)
    return IngestAccepted(jobs=[JobOut.model_validate(j) for j in jobs])


@router.post("/upload", status_code=202, response_model=IngestAccepted, dependencies=[Depends(rate_limit)])
async def upload(file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx)) -> IngestAccepted:
    """Upload a file. Stored under UPLOAD_DIR (shared with workers) and queued; re-uploading
    the same filename replaces the previous version."""
    name = Path(file.filename or "upload").name
    if not re.match(r"^[\w.\- ]{1,200}$", name) or name.startswith("."):
        raise HTTPException(400, "invalid filename")
    try:
        classify_path(Path(name))
    except UnsupportedSource as exc:
        raise HTTPException(415, str(exc)) from exc
    upload_dir = ctx.settings.upload_dir.expanduser().resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid.uuid4().hex}_{name}"
    size = 0
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > ctx.settings.max_upload_bytes:
                handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(413, "file too large")
            handle.write(chunk)
    source_uri = f"upload/{name}"
    jobs = await _create_jobs(ctx, [("file", source_uri, {"path": str(destination), "source_uri": source_uri})])
    return IngestAccepted(jobs=[JobOut.model_validate(j) for j in jobs])


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(
    ctx: AppContext = Depends(get_ctx),
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
) -> list[JobOut]:
    stmt = select(IngestJob).order_by(IngestJob.created_at.desc()).limit(limit)
    if status_filter:
        stmt = stmt.where(IngestJob.status == status_filter)
    async with ctx.sessionmaker() as session:
        return [JobOut.model_validate(j) for j in (await session.scalars(stmt)).all()]


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, ctx: AppContext = Depends(get_ctx)) -> JobOut:
    async with ctx.sessionmaker() as session:
        job = await session.get(IngestJob, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JobOut.model_validate(job)
