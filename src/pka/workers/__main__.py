"""python -m pka.workers --queues ingest,reindex,memory

Run separate pools by queue: e.g. embedding-heavy `--queues ingest,reindex` on a GPU box and
LLM-bound `--queues memory` elsewhere — topic routing makes that a deployment choice."""

import argparse
import asyncio

from pka.core.config import get_settings
from pka.workers.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal Knowledge Agent worker")
    parser.add_argument("--queues", default="ingest,reindex,memory")
    args = parser.parse_args()
    queues = [q.strip() for q in args.queues.split(",") if q.strip()]
    settings = get_settings()
    asyncio.run(run(settings, queues))


if __name__ == "__main__":
    main()
