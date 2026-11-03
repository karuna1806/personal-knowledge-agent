"""Initial schema — §4 of the architecture document, verbatim except the embedding
dimension, which follows EMBED_DIM (the configured model's output size).

Revision ID: 0001
Revises:
"""

from alembic import op
from pka.core.config import get_settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    dim = int(get_settings().embed_dim)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # ─── Corpus ────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE documents (
            id            BIGSERIAL PRIMARY KEY,
            source_type   TEXT NOT NULL,
            source_uri    TEXT NOT NULL,
            title         TEXT,
            content       TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            metadata      JSONB NOT NULL DEFAULT '{}',
            ingested_at   TIMESTAMPTZ,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (source_uri, content_hash)
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE chunks (
            id            BIGSERIAL PRIMARY KEY,
            document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index   INT NOT NULL,
            content       TEXT NOT NULL,
            token_count   INT NOT NULL,
            embedding     VECTOR({dim}) NOT NULL,
            embed_model   TEXT NOT NULL,
            tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            metadata      JSONB NOT NULL DEFAULT '{{}}',
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        )
        """
    )
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute("CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv)")
    op.execute("CREATE INDEX chunks_meta_gin ON chunks USING gin (metadata jsonb_path_ops)")

    # ─── Conversation ──────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE conversations (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title       TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE messages (
            id               BIGSERIAL PRIMARY KEY,
            conversation_id  UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role             TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
            content          TEXT NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # ─── Provenance ────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE citations (
            id           BIGSERIAL PRIMARY KEY,
            message_id   BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            chunk_id     BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            rank         INT NOT NULL,
            score        REAL,
            span_start   INT,
            span_end     INT
        )
        """
    )

    # ─── Memory ────────────────────────────────────────────────
    op.execute(
        f"""
        CREATE TABLE memories (
            id            BIGSERIAL PRIMARY KEY,
            kind          TEXT NOT NULL CHECK (kind IN ('preference','fact','project','entity')),
            subject       TEXT NOT NULL,
            predicate     TEXT NOT NULL,
            object        TEXT NOT NULL,
            confidence    REAL NOT NULL DEFAULT 0.5,
            embedding     VECTOR({dim}),
            source_msg_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,
            superseded_by BIGINT REFERENCES memories(id),
            valid_from    TIMESTAMPTZ NOT NULL DEFAULT now(),
            valid_until   TIMESTAMPTZ,
            access_count  INT NOT NULL DEFAULT 0,
            last_accessed TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE INDEX memories_active ON memories (kind, subject) "
        "WHERE superseded_by IS NULL AND valid_until IS NULL"
    )

    # ─── Observability & eval ──────────────────────────────────
    op.execute(
        """
        CREATE TABLE query_logs (
            id                  BIGSERIAL PRIMARY KEY,
            conversation_id     UUID REFERENCES conversations(id) ON DELETE SET NULL,
            query               TEXT NOT NULL,
            route               TEXT,
            retrieval_ms        INT,
            rerank_ms           INT,
            llm_ms              INT,
            ttft_ms             INT,
            total_ms            INT,
            prompt_tokens       INT,
            completion_tokens   INT,
            model               TEXT,
            chunks_retrieved    INT,
            cache_hit           BOOLEAN DEFAULT false,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE eval_runs (
            id           BIGSERIAL PRIMARY KEY,
            name         TEXT NOT NULL,
            config       JSONB NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE eval_results (
            id                  BIGSERIAL PRIMARY KEY,
            run_id              BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
            question            TEXT NOT NULL,
            expected_chunk_ids  BIGINT[],
            retrieved_chunk_ids BIGINT[],
            answer              TEXT,
            recall_at_k         REAL,
            mrr                 REAL,
            faithfulness        REAL,
            notes               TEXT
        )
        """
    )


def downgrade() -> None:
    for table in (
        "eval_results",
        "eval_runs",
        "query_logs",
        "memories",
        "citations",
        "messages",
        "conversations",
        "chunks",
        "documents",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
