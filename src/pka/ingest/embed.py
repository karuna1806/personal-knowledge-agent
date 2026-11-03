"""Embedding with a Redis cache in front (`emb:{model}:{sha256}` → float32 bytes, 30d TTL).

The cache is also what makes ingest retries cheap: if embedding dies on chunk 400 of 500,
the retry finds chunks 1-399 in the cache and only embeds the rest (§3.5).
"""

import asyncio
import hashlib
import logging
import re
from typing import Protocol

import numpy as np
from redis.asyncio import Redis

from pka.core.metrics import CACHE_REQUESTS, STAGE_LATENCY
from pka.core.telemetry import tracer

log = logging.getLogger(__name__)


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, *, device: str = "", query_prompt: str = "", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device or None)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(dim_fn())
        prompts = getattr(self._model, "prompts", {}) or {}
        self._query_prompt = query_prompt if query_prompt in prompts else None
        if query_prompt and self._query_prompt is None:
            log.warning("embedding model %s has no prompt %r; queries embed without one", model_name, query_prompt)
        self._batch_size = batch_size

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        kwargs = {"prompt_name": self._query_prompt} if is_query and self._query_prompt else {}
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashEncoder:
    """Deterministic bag-of-words hashing encoder. No model download; used by tests and as a
    smoke-test backend (EMBED_MODEL=hash). Similar texts share tokens → similar vectors."""

    name = "hash"

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


def make_encoder(model_name: str, dim: int, *, device: str = "", query_prompt: str = "",
                 batch_size: int = 16) -> Encoder:  # fmt: skip
    if model_name == "hash":
        return HashEncoder(dim)
    encoder = SentenceTransformerEncoder(
        model_name, device=device, query_prompt=query_prompt, batch_size=batch_size
    )
    if encoder.dim != dim:
        raise ValueError(
            f"{model_name} produces {encoder.dim}-dim vectors but EMBED_DIM={dim}; "
            "the schema's VECTOR(n) must match the model"
        )
    return encoder


class Embedder:
    """Async facade: cache lookup, then the encoder in a worker thread. A single lock
    serialises model calls — torch on MPS/CPU is not safe to drive from many threads."""

    def __init__(self, encoder: Encoder, redis: Redis | None, ttl_s: int):
        self.encoder = encoder
        self.model_name = encoder.name
        self.dim = encoder.dim
        self._redis = redis
        self._ttl_s = ttl_s
        self._lock = asyncio.Lock()

    def _key(self, text: str, is_query: bool) -> str:
        digest = hashlib.sha256((("q\x00" if is_query else "d\x00") + text).encode()).hexdigest()
        return f"emb:{self.model_name}:{digest}"

    async def _cache_get(self, keys: list[str]) -> list[bytes | None]:
        if self._redis is None:
            return [None] * len(keys)
        try:
            return await self._redis.mget(keys)
        except Exception as exc:  # Redis is a cache: losing it costs latency, not correctness
            log.warning("embedding cache read failed: %s", exc)
            return [None] * len(keys)

    async def _cache_set(self, items: dict[str, bytes]) -> None:
        if self._redis is None or not items:
            return
        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                for key, value in items.items():
                    pipe.set(key, value, ex=self._ttl_s)
                await pipe.execute()
        except Exception as exc:
            log.warning("embedding cache write failed: %s", exc)

    async def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with tracer.start_as_current_span("embed") as span:
            span.set_attribute("embed.count", len(texts))
            keys = [self._key(t, is_query) for t in texts]
            cached = await self._cache_get(keys)
            out = np.zeros((len(texts), self.dim), dtype=np.float32)
            missing: list[int] = []
            for i, blob in enumerate(cached):
                if blob is not None and len(blob) == self.dim * 4:
                    out[i] = np.frombuffer(blob, dtype=np.float32)
                else:
                    missing.append(i)
            hits = len(texts) - len(missing)
            if hits:
                CACHE_REQUESTS.labels("emb", "hit").inc(hits)
            if missing:
                CACHE_REQUESTS.labels("emb", "miss").inc(len(missing))
                with STAGE_LATENCY.labels("embed").time():
                    async with self._lock:
                        vectors = await asyncio.to_thread(
                            self.encoder.encode, [texts[i] for i in missing], is_query=is_query
                        )
                for j, i in enumerate(missing):
                    out[i] = vectors[j]
                await self._cache_set({keys[i]: out[i].tobytes() for i in missing})
            span.set_attribute("embed.cache_hits", hits)
            return out

    async def embed_query(self, text: str) -> np.ndarray:
        return (await self.embed([text], is_query=True))[0]
