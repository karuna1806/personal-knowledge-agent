"""Cross-encoder rerank: retrieve 20 cheaply, score each (query, chunk) pair jointly, keep 5.
With a small local model and a 4-8k context, precision here *is* answer quality (§3.9)."""

import asyncio
import time
from typing import Protocol

from pka.agent.retriever import RetrievedChunk
from pka.core.metrics import STAGE_LATENCY
from pka.core.telemetry import tracer
from pka.ingest.chunkers import embedding_text


class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> tuple[list[RetrievedChunk], int]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "", max_length: int = 512):
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        if not device:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # fp16 halves rerank latency on GPU/MPS (≈800→440 ms for 20 pairs on an M-series Mac)
        # with the same ranking; on CPU it would be slower, so CPU stays fp32.
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if device.startswith(("cuda", "mps")) else {}
        self._model = CrossEncoder(model_name, device=device, max_length=max_length, **kwargs)
        self._lock = asyncio.Lock()

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> tuple[list[RetrievedChunk], int]:
        if not chunks:
            return [], 0
        started = time.perf_counter()
        with tracer.start_as_current_span("rerank") as span, STAGE_LATENCY.labels("rerank").time():
            span.set_attribute("rerank.candidates", len(chunks))
            pairs = [(query, embedding_text(c.content, c.metadata, c.title)) for c in chunks]
            async with self._lock:
                scores = await asyncio.to_thread(self._score, pairs)
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = score
        ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_k]
        return ranked, int((time.perf_counter() - started) * 1000)


class PassthroughReranker:
    """Rerank disabled: keep fused (RRF) order. Used for the with/without-rerank experiment."""

    model_name = "none"

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> tuple[list[RetrievedChunk], int]:
        return chunks[:top_k], 0
