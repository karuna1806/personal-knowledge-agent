# ADR 0008: Streaming: SSE over WebSocket

- **Status:** accepted
- **Source:** architecture document §3.8

**Chosen:** Server-Sent Events

**Why:** Token streaming is unidirectional — server to client. SSE is plain HTTP: it works through proxies and load balancers without upgrade negotiation, browsers reconnect automatically via `EventSource`, and my auth middleware applies unchanged. WebSocket buys bidirectional communication I don't need, in exchange for connection state management, heartbeats, and a separate auth path.

**What would change my mind:** if I added collaborative features or client-initiated interrupts mid-generation, WebSocket becomes correct. Cancellation alone doesn't force it — closing the SSE connection is a fine cancel signal.
