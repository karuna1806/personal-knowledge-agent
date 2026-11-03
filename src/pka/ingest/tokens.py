"""Token counting with the embedding model's own tokenizer, so chunk budgets match what the
embedder actually sees. Falls back to a ~4 chars/token estimate when no tokenizer loads
(e.g. the hash embedder used in tests)."""

from collections.abc import Callable
from functools import lru_cache

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


@lru_cache(maxsize=4)
def get_token_counter(model_name: str) -> TokenCounter:
    if model_name == "hash":
        return estimate_tokens
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return estimate_tokens

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
