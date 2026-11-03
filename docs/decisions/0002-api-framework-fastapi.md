# ADR 0002: API framework: FastAPI

- **Status:** accepted
- **Source:** architecture document §3.2

**Chosen:** FastAPI

**Rejected:** Django REST Framework, Flask, Litestar

| | FastAPI | Django REST | Flask | Litestar |
|---|---|---|---|---|
| Native async | Yes | Partial (ORM async is limited) | Bolted on | Yes |
| Streaming (SSE) | First-class | Awkward | Manual | First-class |
| Request validation | Pydantic, built in | Serializers, verbose | Manual | Pydantic |
| OpenAPI docs | Automatic | Plugin | Plugin | Automatic |
| Right-sized for this | Yes | Overkill (admin, auth, templates unused) | Too thin | Yes, but smaller ecosystem |

**Why FastAPI over Litestar specifically** (this is the question a sharp interviewer asks, because Litestar is the credible modern alternative): Litestar is arguably better designed — cleaner DI, better layered architecture. I chose FastAPI for ecosystem gravity: more integration examples for the LLM tooling I'm using, and a much larger body of production knowledge to draw on when something breaks at 1am. That's a deliberate tradeoff of elegance for operational safety.

**Why not Django:** I'd be importing an ORM, admin, template engine, and auth system to use roughly 15% of it. Django wins when you need the batteries; here the batteries are dead weight, and its async ORM support is still the weakest part of the stack.

**Key FastAPI features actually used:** `StreamingResponse` for SSE token streaming, dependency injection for DB sessions and auth, Pydantic v2 models for every request/response boundary, `BackgroundTasks` only for fire-and-forget logging (real async work goes to the queue — see §3.5).
