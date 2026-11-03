# ADR 0007: Tooling: MCP (Model Context Protocol)

- **Status:** accepted
- **Source:** architecture document §3.7

**Chosen:** MCP for tool integration, with a hard constraint on how the local model uses it

MCP is an open protocol for exposing tools and data sources to LLM applications over a standard JSON-RPC interface. The value for this project: instead of writing bespoke Python functions for every capability and hardcoding the schema plumbing, I run MCP servers and my agent discovers their tools at runtime.

**Servers I'd wire up, and why each:**

| Server | Capability | Why it fits a personal knowledge agent |
|---|---|---|
| **filesystem** | Read/list local files | Live access to files not yet ingested — "what's in my current project dir" |
| **git** | Log, diff, blame, show | "What did I change in this repo last week" — temporal code questions RAG can't answer from static chunks |
| **postgres** | Read-only SQL | Structured queries over my own metadata — "how many notes did I write in March" |
| **fetch** | Retrieve a URL | Pull in a page on demand and optionally queue it for ingestion |

**The critical honest constraint — and this is a strong interview point:** a quantized 7B model is *not reliable* at multi-step tool selection. It will hallucinate tool names, malform arguments, and loop. Pretending otherwise produces a demo that breaks on stage.

**How the architecture handles that instead of ignoring it:**

1. **Constrained decoding for tool arguments.** Both llama.cpp (GBNF grammars) and vLLM (guided decoding via outlines/xgrammar) can force output to conform to a JSON schema. Malformed tool calls become *structurally impossible* rather than something I validate and retry. This is the single highest-leverage technique for making small models usable for structured output.
2. **Router-first, not agent-first.** A single classification step picks one path — `retrieve` | `tool` | `direct` — rather than an open ReAct loop. One decision, validated against a fixed enum, is dramatically more reliable than an autonomous multi-step plan.
3. **Hard iteration cap.** Max 3 tool calls per query, then synthesize with whatever was gathered. No unbounded loops.
4. **Every tool result is context, never an action.** All four servers above are read-only. The agent cannot write files, push commits, or mutate the DB. Blast radius: zero.

**Why MCP over just writing Python functions:** the tool layer becomes portable and independently testable. I can point any MCP-capable client at the same servers, and I can test a server without running the agent. It also means my tools aren't coupled to my framework choice.
