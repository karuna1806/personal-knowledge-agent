# Evaluation

`python -m evals.run` scores the golden set ([`evals/golden_set.yaml`](../../evals/golden_set.yaml)).
Every run is stored in `eval_runs` / `eval_results` and written up here as `<name>.md`, with
per-category and per-question tables.

Retrieval is measured independently of generation (Recall@k, MRR, nDCG@5 over the reranked
list) so a regression can be pinned on the stage that caused it. Ground truth is a set of
*locators* — `(source, phrase)` — rather than chunk ids, so labels survive re-ingestion and
work across chunking strategies.

**Setup for the numbers below:** Apple M5 (24 GB); Qwen3-Embedding-0.6B; bge-reranker-base;
`gemma4:12b` Q4_K_M via Ollama (4096 context, thinking off, temperature 0); corpus = the
architecture document + three early RAG scripts (**53 chunks**); 34 questions (28 answerable:
15 lookup, 7 exact-identifier, 4 synthesis, 2 code; 6 negative).

> **Read these with the sample size in mind.** With 53 chunks retrieval is easy, and one
> question flipping moves a category average by 0.14–0.5. The harness is the deliverable;
> grow the corpus and golden set (50–80 real questions) before trusting small gaps.

## Retrieval experiments

| Run | Chunker | Retrieval | Rerank | R@5 | R@20 | MRR | nDCG@5 | Identifier R@5 | Identifier MRR |
|---|---|---|---|---|---|---|---|---|---|
| [baseline-retrieval](baseline-retrieval.md) | structure | hybrid | ✓ | **0.982** | 1.000 | **0.905** | **0.914** | **1.000** | **0.929** |
| [vector-only](vector-only.md) | structure | vector | ✓ | 0.964 | 0.964 | 0.905 | 0.906 | 1.000 | 0.929 |
| [lexical-only](lexical-only.md) | structure | lexical | ✓ | 0.964 | 1.000 | 0.905 | 0.906 | 1.000 | 0.929 |
| [hybrid-no-rerank](hybrid-no-rerank.md) | structure | hybrid | – | 0.929 | 1.000 | 0.767 | 0.796 | 0.857 | 0.604 |
| [vector-only-no-rerank](vector-only-no-rerank.md) | structure | vector | – | 0.893 | 0.964 | 0.767 | 0.789 | 0.714 | 0.565 |
| [fixed-chunks](fixed-chunks.md) | fixed 1500/200 | hybrid | ✓ | 0.946 | 1.000 | 0.887 | 0.887 | 1.000 | 0.833 |
| [fixed-chunks-no-rerank](fixed-chunks-no-rerank.md) | fixed 1500/200 | hybrid | – | 0.875 | 1.000 | 0.815 | 0.810 | 0.714 | 0.759 |

What the hypotheses from the architecture (§6) look like against these numbers:

| Hypothesis | Result |
|---|---|
| Hybrid substantially improves exact-identifier queries | **Supported without the reranker, masked with it (small n).** Without rerank, identifier R@5 0.714 → 0.857 and MRR 0.565 → 0.604 going vector → hybrid. With rerank on top, vector-only already reaches identifier R@5 1.000 on this corpus — the cross-encoder recovers what the bi-encoder ranked low. Hybrid's remaining edge is recall depth: vector-only misses one (synthesis) locator even at k=20; hybrid finds all. |
| The reranker earns its latency | **Supported.** Hybrid MRR 0.767 → 0.905, nDCG@5 0.796 → 0.914; identifier R@5 0.857 → 1.000. Cost: ~210 ms warm for 20 pairs (fp16 on MPS; ~440 ms cold, ~800 ms fp32). |
| Structure-aware chunking beats fixed-size | **Mostly supported, not uniformly.** Overall R@5 0.982 vs 0.946 (with rerank), 0.929 vs 0.875 (without). But fixed chunks *won* on lookup MRR with rerank (1.000 vs 0.933) and lost synthesis R@5 (0.625 vs 0.875). |
| Lexical alone is weak | **Not on this corpus.** With the reranker on top, lexical-only matched hybrid on identifiers and lookups. The corpus is one well-written document whose vocabulary matches the questions; expect this to change on messier notes. |
| Truncating reranker input is a cheap speed-up | **Rejected.** [`rerank-max-length-256`](rerank-max-length-256.md): CPU rerank 5.9 s → 2.4 s, but R@5 0.982 → 0.839 and identifier R@5 1.000 → 0.571 — *worse than no reranker*. Chunks are up to 512 tokens; cutting them at 256 hides the answer from the cross-encoder. Kept at 512. |

## Generation

[`baseline-generation`](baseline-generation.md) — the default configuration (structure chunks,
hybrid, rerank, top 5), final prompts, temperature 0:

| Metric | Value | What it means |
|---|---|---|
| Refusal accuracy (negatives) | **1.000** (6/6) | every out-of-corpus question got "I don't know based on the available sources." |
| False refusal rate (answerable) | **0.000** (0/28) | never refused when the answer was in the corpus |
| Answers with ≥1 valid citation | **1.000** | |
| Invalid citation rate | **0.000** | no `[n]` pointing at a source that wasn't in the prompt |
| Faithfulness (LLM-judged) | 1.000 | **self-judged** by the same 12B model — treat as an upper bound. A manual spot-check of 7 answers (L03, L08, I06, S01, S03, C02, N02) against the source found no unsupported claims. |
| TTFT p50 / p95 | 11.2 s / 18.5 s | no router call in the harness; this is retrieval + rerank + LLM prefill of a ~2–2.7k-token prompt |

**Latency, decomposed.** The pipeline before the LLM is cheap natively on the M5: embed ~50 ms,
vector + lexical search 5–30 ms, rerank ~210 ms (fp16, MPS). The router adds ~1.6 s. The rest is
the LLM: `gemma4:12b` prefills at ~290–540 tok/s on this machine, so a grounded prompt costs
5–9 s before the first token, then generates at ~15 tok/s. The architecture's < 2 s TTFT target is
not reachable with a 12B model on this hardware; the levers are a smaller model, fewer/smaller
chunks (`FINAL_K`, `CHUNK_MAX_TOKENS` — measure the recall cost here first), or GPU serving.

Inside Docker on macOS there is no GPU access, so the embedder and cross-encoder run on CPU:
rerank takes ~5 s per query there versus ~0.2 s natively. On a Mac, run the API and worker
natively (README → *Running natively*), or set `RERANK_ENABLED=false` (R@5 0.929 vs 0.982).

## Reproducing

```bash
# retrieval-only, against the live index
python -m evals.run --name vector-only --mode vector
python -m evals.run --name hybrid-no-rerank --no-rerank

# chunking experiments build a separate index from ./corpus, so the live DB is untouched
python -m evals.run --name fixed-chunks --build-index --chunker fixed \
    --db postgresql+asyncpg://pka:pka@localhost:5433/pka_eval_fixed

# generation + LLM-judged faithfulness
python -m evals.run --name baseline-generation --judge
```

Not yet run (need hardware or models not available here): 3B vs 7B vs 14B, Q4 vs Q8, and
Ollama vs vLLM under concurrency. `--judge` uses the answering model as its own judge; a
larger judge model is the obvious upgrade.
