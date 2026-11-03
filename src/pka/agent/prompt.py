"""Prompt assembly with a token budget. Two context sources are kept visibly separate (§5):
numbered <sources> (evidence, citable) and <user_memory> (possibly-stale background, never
cited) — so a bad answer can be attributed to bad retrieval vs bad memory."""

from dataclasses import dataclass

from pka.ingest.tokens import TokenCounter
from pka.llm.client import Message

REFUSAL = "I don't know based on the available sources."

RULES_GROUNDED = f"""You are a personal knowledge assistant answering questions about the user's own notes, documents and code.

Rules:
1. Use ONLY the numbered sources inside <sources>. Do not use outside knowledge.
2. Cite every factual claim with its source number in square brackets right after the claim, e.g. "HNSW needs no rebuilds [2]." Use several markers if several sources support it: [1][3].
3. <user_memory> holds things the user told you earlier; it may be stale. Use it to interpret the question. If the question is about the user themselves and only <user_memory> answers it, answer from it and say so ("From what you've told me, ...") — without a citation number. If it conflicts with a source, the source wins.
4. If neither the sources nor <user_memory> contain the answer, reply exactly: "{REFUSAL}" Then, in one sentence, say what is missing. Do not guess.
5. Be concise and direct. Prefer the user's own terminology from the sources."""

RULES_TOOL = f"""You are a personal knowledge assistant. Tools were run to gather live data for the user's question.

Rules:
1. Use ONLY the numbered tool results inside <sources>. Do not use outside knowledge.
2. Cite claims with the source number in square brackets, e.g. [1].
3. If the results do not answer the question (or a tool errored), say "{REFUSAL}" and briefly say why.
4. <user_memory> is possibly-stale background about the user; never cite it with a number.
5. Be concise."""

RULES_DIRECT = f"""You are a personal knowledge assistant. This message needs no document lookup: a greeting, thanks, the user telling you something about themselves, or a request about this conversation.

Rules:
1. Respond briefly and naturally. If the user is telling you a preference or fact about themselves, acknowledge it in one short sentence (it is saved to memory automatically) — do not say you don't know.
2. Do not state facts about the world or the user's work from general knowledge. Only if the message actually ASKS for such facts, say "{REFUSAL}" and suggest asking it as a question about their notes.
3. <user_memory> is possibly-stale background about the user."""

# Tokenizer mismatch margin: we count with the embedding model's tokenizer, the LLM's differs.
SAFETY = 0.85


@dataclass
class Source:
    """One numbered entry in <sources>: a retrieved chunk or a tool result."""

    n: int
    kind: str  # chunk | tool
    label: str
    content: str
    chunk_id: int | None = None
    score: float | None = None
    source_uri: str | None = None
    title: str | None = None


@dataclass
class BuiltPrompt:
    messages: list[Message]
    sources: list[Source]
    prompt_tokens_est: int
    dropped_sources: int
    dropped_history: int


def _memory_block(memories: list) -> str:
    if not memories:
        return ""
    lines = [
        f"- ({m.kind}) {m.subject} {m.predicate.replace('_', ' ')}: {m.object} "
        f"[since {m.valid_from:%Y-%m-%d}]"
        for m in memories
    ]
    return "<user_memory note=\"possibly stale; background only; never cite\">\n" + "\n".join(lines) + "\n</user_memory>"


def _source_block(sources: list[Source]) -> str:
    if not sources:
        return "<sources>\n(no sources were found)\n</sources>"
    body = "\n\n".join(f"[{s.n}] {s.label}\n{s.content}" for s in sources)
    return f"<sources>\n{body}\n</sources>"


def build_prompt(
    *,
    route: str,
    question: str,
    candidates: list[Source],
    memories: list,
    history: list[Message],
    count: TokenCounter,
    context_tokens: int,
    max_answer_tokens: int,
) -> BuiltPrompt:
    """Fill the context window in priority order: rules + question + memory (always), then
    sources in rank order, then as much recent history as still fits. Sources outrank history
    because a follow-up is already resolved into `question` by the router."""
    rules = {"retrieve": RULES_GROUNDED, "tool": RULES_TOOL}.get(route, RULES_DIRECT)
    budget = int((context_tokens - max_answer_tokens) * SAFETY)
    memory = _memory_block(memories)
    fixed = rules + memory + question
    remaining = budget - count(fixed) - 40  # chat-template overhead

    chosen: list[Source] = []
    dropped_sources = 0
    if route in ("retrieve", "tool"):
        for candidate in candidates:
            cost = count(candidate.label) + count(candidate.content) + 8
            if cost <= remaining:
                chosen.append(candidate)
                remaining -= cost
            else:
                dropped_sources += 1
    # Renumber densely so the model sees [1]..[n] with no gaps.
    for i, source in enumerate(chosen, start=1):
        source.n = i

    kept_history: list[Message] = []
    for message in reversed(history):
        cost = count(message["content"]) + 6
        if cost > remaining:
            break
        kept_history.insert(0, message)
        remaining -= cost
    # A transcript must start with a user turn for most chat templates.
    while kept_history and kept_history[0]["role"] != "user":
        kept_history.pop(0)

    system = rules + ("\n\n" + memory if memory else "")
    if route in ("retrieve", "tool"):
        user = f"{_source_block(chosen)}\n\nQuestion: {question}"
    else:
        user = question
    messages: list[Message] = [{"role": "system", "content": system}, *kept_history, {"role": "user", "content": user}]
    return BuiltPrompt(
        messages=messages,
        sources=chosen,
        prompt_tokens_est=budget - remaining,
        dropped_sources=dropped_sources,
        dropped_history=len(history) - len(kept_history),
    )
