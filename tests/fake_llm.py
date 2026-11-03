"""A deterministic OpenAI-compatible server, so integration tests exercise the real
LLMClient (HTTP, streaming, response_format) without a GPU.

Behaviour:
- json_schema "route": "retrieve", unless the message says "hello" → "direct";
  standalone_query echoes the latest message when the schema asks for it.
- json_schema "memories": "I prefer X" → preference triple prefers_tool=X.
- streaming: an answer citing [1] built from the first source's first line, or the
  refusal when the prompt has no sources.
"""

import json
import re
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
calls: list[dict] = []


def _json_reply(body: dict) -> dict:
    schema = body["response_format"]["json_schema"]
    name = schema["name"]
    last = body["messages"][-1]["content"]
    if name == "route":
        latest = last.split("Latest message:")[-1].strip()
        out = {"route": "direct" if "hello" in latest.lower() else "retrieve"}
        if "standalone_query" in schema["schema"]["properties"]:
            out["standalone_query"] = latest + " (standalone)"
        return out
    if name == "memories":
        said = last.split("<<<")[1].split(">>>")[0]
        m = re.search(r"I prefer ([\w\-]+)", said)
        if not m:
            return {"memories": []}
        return {"memories": [{"kind": "preference", "subject": "user", "predicate": "prefers_tool",
                              "object": m.group(1).strip(), "confidence": 0.9}]}  # fmt: skip
    return {}


def _answer(body: dict) -> str:
    user = body["messages"][-1]["content"]
    m = re.search(r"\[1\] [^\n]*\n([^\n]+)", user)
    if "<sources>" in user and m:
        return f"From the notes: {m.group(1).strip()[:80]} [1]."
    return "I don't know based on the available sources."


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "fake", "object": "model", "created": 0, "owned_by": "test"}]}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    calls.append(body)
    base = {"id": "x", "created": int(time.time()), "model": body["model"]}
    if body.get("response_format", {}).get("type") == "json_schema":
        content = json.dumps(_json_reply(body))
        return JSONResponse({**base, "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})  # fmt: skip
    text = _answer(body)
    if not body.get("stream"):
        return JSONResponse({**base, "object": "chat.completion", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})  # fmt: skip

    async def stream():
        for word in re.findall(r"\S+\s*", text):
            chunk = {**base, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]}  # fmt: skip
            yield f"data: {json.dumps(chunk)}\n\n"
        done = {**base, "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}  # fmt: skip
        yield f"data: {json.dumps(done)}\n\n"
        usage = {**base, "object": "chat.completion.chunk", "choices": [],
                 "usage": {"prompt_tokens": 100, "completion_tokens": len(text.split()), "total_tokens": 0}}  # fmt: skip
        yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def start() -> tuple[str, uvicorn.Server]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}/v1", server
