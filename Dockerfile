FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

# git: the git MCP server shells out to it (read-only subcommands).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch by default: the generic wheel pulls ~2 GB of CUDA libraries that a CPU
# container never uses. docker-compose.gpu.yml overrides this with a CUDA index.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX_URL} torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Bake the tree-sitter grammars into the image so code ingestion never needs network.
RUN python -c "import tree_sitter_language_pack as t; from pka.ingest.loaders.base import CODE_LANGUAGES; t.download(sorted(CODE_LANGUAGES))"

COPY alembic.ini mcp_servers.yaml ./
COPY alembic ./alembic
COPY evals ./evals

ENV MCP_CONFIG=/app/mcp_servers.yaml \
    LOG_JSON=true
EXPOSE 8000 9101
CMD ["uvicorn", "pka.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
