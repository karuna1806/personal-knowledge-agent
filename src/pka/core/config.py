"""All configuration in one place, read from the environment (and `.env`)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Infrastructure ────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://pka:pka@localhost:5433/pka"
    readonly_database_url: str = "postgresql://pka_ro:pka_ro@localhost:5433/pka"
    redis_url: str = "redis://localhost:6379/0"
    amqp_url: str = "amqp://pka:pka@localhost:5672/"

    # ── Gateway ───────────────────────────────────────────────────
    # Single API key (§11: no multi-user auth). Empty string disables auth — dev only.
    api_key: str = "dev-key"
    rate_limit_per_minute: int = 60
    upload_dir: Path = Path("./data/uploads")
    corpus_root: Path = Path("./corpus")
    max_upload_bytes: int = 50 * 1024 * 1024

    # ── LLM (OpenAI-compatible: Ollama or vLLM) ───────────────────
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "gemma4:12b"
    # Ollama's gemma4/qwen3 "think" before answering by default, which costs ~25s per call.
    # "none" turns it off. Set to "" for servers that reject the parameter.
    llm_reasoning_effort: str = "none"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 768
    # Must match the server's real context window. Ollama defaults to 4096 and, through the
    # OpenAI-compatible endpoint, silently drops the *start* of an over-long prompt (i.e. the
    # system prompt). Raise both together: OLLAMA_CONTEXT_LENGTH=8192 and LLM_CONTEXT_TOKENS=8192.
    llm_context_tokens: int = 4096
    llm_timeout_s: float = 120.0

    # ── Embeddings ────────────────────────────────────────────────
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embed_dim: int = 1024
    embed_device: str = ""  # "" = auto (cuda > mps > cpu)
    embed_batch_size: int = 16
    embed_query_prompt: str = "query"  # sentence-transformers prompt_name for queries; "" = none

    # ── Chunking ──────────────────────────────────────────────────
    chunker: str = "structure"  # structure | fixed  (fixed = baseline for the §6 experiment)
    chunk_max_tokens: int = 512
    chunk_fixed_chars: int = 1500
    chunk_fixed_overlap: int = 200

    # ── Retrieval ─────────────────────────────────────────────────
    retrieval_mode: str = "hybrid"  # hybrid | vector | lexical
    vector_k: int = 20
    lexical_k: int = 20
    fused_k: int = 20
    final_k: int = 5
    rrf_k: int = 60
    hnsw_ef_search: int = 64
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    # Tokens per (query, chunk) pair. Lowering it is NOT a free speed-up: 256 cut CPU rerank
    # from ~5.9 s to ~2.4 s but dropped Recall@5 0.982 → 0.839 (worse than no reranker), because
    # chunks are up to CHUNK_MAX_TOKENS=512 long. See docs/evals/rerank-max-length-256.md.
    rerank_max_length: int = 512

    # ── Caches (Redis) ────────────────────────────────────────────
    embed_cache_ttl_s: int = 30 * 24 * 3600
    query_cache_ttl_s: int = 3600
    session_ttl_s: int = 24 * 3600
    session_turns: int = 6

    # ── Memory ────────────────────────────────────────────────────
    memory_enabled: bool = True
    memory_top_k: int = 5
    memory_min_similarity: float = 0.45
    memory_w_similarity: float = 0.6
    memory_w_recency: float = 0.2
    memory_w_access: float = 0.2
    memory_half_life_days: float = 30.0

    # ── Tools (MCP) ───────────────────────────────────────────────
    mcp_enabled: bool = True
    mcp_config: Path = Path("mcp_servers.yaml")
    mcp_max_tool_calls: int = 3
    mcp_tool_timeout_s: float = 20.0
    mcp_result_max_chars: int = 6000

    # ── Queue ─────────────────────────────────────────────────────
    queue_max_attempts: int = 3
    queue_retry_delay_ms: int = 5000

    # ── Observability ─────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = True
    otel_exporter_otlp_endpoint: str = ""
    service_name: str = Field(default="pka-api")
    worker_metrics_port: int = 9101


@lru_cache
def get_settings() -> Settings:
    return Settings()
