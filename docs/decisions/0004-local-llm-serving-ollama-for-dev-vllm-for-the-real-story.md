# ADR 0004: Local LLM serving: Ollama for dev, vLLM for the real story

- **Status:** accepted
- **Source:** architecture document §3.4

**Chosen:** Both, behind one interface

**Why both:** They optimize for different things and I want to be able to talk about the difference.

| | Ollama | vLLM |
|---|---|---|
| Setup | One command | Real config, CUDA required |
| Backend | llama.cpp / GGUF | PyTorch, PagedAttention |
| Quantization | Excellent (runs on CPU/Mac) | Limited (AWQ/GPTQ, GPU only) |
| Batching | Basic | Continuous batching |
| Throughput under load | Poor | Strong |
| Good for | Dev, demo, laptops | Serving, benchmarks |

**The architectural move that makes this cheap:** both expose an **OpenAI-compatible `/v1/chat/completions` endpoint.** So my agent core talks to one interface and the backend is a config value:

```python
# llm/client.py — one client, swappable backend
class LLMClient:
    def __init__(self, base_url: str, model: str):
        self._client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
        self._model = model

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        ...
```

`LLM_BASE_URL=http://ollama:11434/v1` on my laptop, `http://vllm:8000/v1` on a GPU box. Same code path.

**Why this is worth doing rather than picking one:** it lets me run a real benchmark — same prompts, same corpus, measure tokens/sec and time-to-first-token under concurrent load on both. "I benchmarked my inference layer and here are the numbers" is a substantially stronger interview answer than "I used Ollama."

**Model choice:** start with a 7–8B instruct model (Llama 3.1 8B, Mistral 7B, or Qwen2.5 7B) at Q4_K_M quantization. Rationale: fits in ~5–6GB VRAM quantized, and — more important for RAG — has adequate instruction-following to stay grounded in provided context. Sub-3B models drift off-context and start answering from parametric memory, which is exactly the failure mode I'm trying to engineer against.

**Note on tool calling:** open 7B models are meaningfully worse at structured tool calling than frontier models. This is a real constraint, not a footnote — see §3.7 for how the architecture accommodates it.
