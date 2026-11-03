"""Ingest job tracking (not in §4): lets clients poll an async ingest and ties DLQ entries
back to a source. Also indexes chunks.embed_model for stale-row reindexing and
memories(subject, predicate) for supersession lookups.

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ingest_jobs (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            kind         TEXT NOT NULL CHECK (kind IN ('file','url','reindex')),
            target       TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'queued'
                         CHECK (status IN ('queued','running','done','skipped','failed','dead')),
            attempts     INT NOT NULL DEFAULT 0,
            document_id  BIGINT REFERENCES documents(id) ON DELETE SET NULL,
            chunks       INT,
            error        TEXT,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ingest_jobs_status ON ingest_jobs (status, created_at DESC)")
    op.execute("CREATE INDEX chunks_embed_model ON chunks (embed_model)")
    op.execute("CREATE INDEX documents_source_uri ON documents (source_uri)")
    op.execute(
        "CREATE INDEX memories_active_sp ON memories (subject, predicate) "
        "WHERE superseded_by IS NULL AND valid_until IS NULL"
    )
    op.execute("CREATE INDEX citations_message ON citations (message_id)")
    op.execute("CREATE INDEX messages_conversation ON messages (conversation_id, id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS messages_conversation")
    op.execute("DROP INDEX IF EXISTS citations_message")
    op.execute("DROP INDEX IF EXISTS memories_active_sp")
    op.execute("DROP INDEX IF EXISTS documents_source_uri")
    op.execute("DROP INDEX IF EXISTS chunks_embed_model")
    op.execute("DROP TABLE IF EXISTS ingest_jobs")
