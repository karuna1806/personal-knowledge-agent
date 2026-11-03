from pka.agent.router import route_query, route_schema
from pka.agent.tools import run_tool_loop
from pka.mcp.client import ToolResult, ToolSpec


class ScriptedLLM:
    """Returns queued JSON replies for complete_json, recording each schema it was given."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.schemas = []

    async def complete_json(self, messages, schema, *, name="output", max_tokens=512):
        self.schemas.append((name, schema))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeTools:
    def __init__(self):
        self.tools = {
            "fs__list": ToolSpec("fs__list", "fs", "list", "List a dir", {
                "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
            "git__log": ToolSpec("git__log", "git", "log", "Commit history", {
                "type": "object", "properties": {"since": {"type": "string"}}}),
        }  # fmt: skip
        self.calls = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        return ToolResult(f"output of {name}", False)


async def test_router_offers_tool_only_when_tools_exist():
    llm = ScriptedLLM([{"route": "retrieve"}, {"route": "tool"}])
    await route_query(llm, "q", [], tool_lines=[])
    await route_query(llm, "q", [], tool_lines=["git__log(since): history"])
    assert llm.schemas[0][1]["properties"]["route"]["enum"] == ["retrieve", "direct"]
    assert llm.schemas[1][1]["properties"]["route"]["enum"] == ["retrieve", "tool", "direct"]
    assert "standalone_query" not in llm.schemas[0][1]["properties"]  # no history → no rewrite


async def test_router_rewrites_follow_ups_and_falls_back_safely():
    history = [{"role": "user", "content": "tell me about RabbitMQ"}, {"role": "assistant", "content": "..."}]
    llm = ScriptedLLM([{"route": "retrieve", "standalone_query": "What is RabbitMQ's DLQ?"}])
    d = await route_query(llm, "and its DLQ?", history, tool_lines=[])
    assert d.standalone_query == "What is RabbitMQ's DLQ?" and not d.fallback
    assert "standalone_query" in llm.schemas[0][1]["required"]

    broken = ScriptedLLM([TimeoutError("llm down")])
    d = await route_query(broken, "q", [], tool_lines=[])
    assert (d.route, d.standalone_query, d.fallback) == ("retrieve", "q", True)

    off_enum = ScriptedLLM([{"route": "tool"}])  # tool not offered → treated as failure
    assert (await route_query(off_enum, "q", [], tool_lines=[])).fallback


def test_route_schema_shape():
    s = route_schema(["retrieve", "direct"], rewrite=True)
    assert s["required"] == ["route", "standalone_query"] and s["additionalProperties"] is False


async def test_tool_loop_calls_then_answers():
    llm = ScriptedLLM([
        {"action": "call_tool", "tool": "git__log"}, {"since": "1 week ago"},
        {"action": "answer", "tool": "git__log"},
    ])  # fmt: skip
    tools = FakeTools()
    obs = await run_tool_loop(llm, tools, "what changed last week?", max_calls=3)
    assert tools.calls == [("git__log", {"since": "1 week ago"})]
    assert [o.tool for o in obs] == ["git__log"]
    # step 0 may not "answer"; later steps may; tool names are an enum
    assert llm.schemas[0][1]["properties"]["action"]["enum"] == ["call_tool"]
    assert llm.schemas[2][1]["properties"]["action"]["enum"] == ["call_tool", "answer"]
    assert llm.schemas[0][1]["properties"]["tool"]["enum"] == ["fs__list", "git__log"]
    # arguments are generated under the tool's own schema
    assert llm.schemas[1][0] == "tool_args" and llm.schemas[1][1]["properties"] == {"since": {"type": "string"}}


async def test_tool_loop_rejects_invalid_arguments_and_caps_calls():
    llm = ScriptedLLM([
        {"action": "call_tool", "tool": "fs__list"}, {},                  # missing required "path"
        {"action": "call_tool", "tool": "fs__list"}, {"path": "notes"},
        {"action": "call_tool", "tool": "git__log"}, {"since": "today"},
    ])  # fmt: skip
    tools = FakeTools()
    obs = await run_tool_loop(llm, tools, "q", max_calls=3)
    assert [o.is_error for o in obs] == [True, False, False]
    assert "could not build valid arguments" in obs[0].output
    assert tools.calls == [("fs__list", {"path": "notes"}), ("git__log", {"since": "today"})]
    assert llm.replies == []  # exactly max_calls steps were taken


async def test_tool_loop_stops_on_identical_repeat():
    llm = ScriptedLLM([
        {"action": "call_tool", "tool": "fs__list"}, {"path": "."},
        {"action": "call_tool", "tool": "fs__list"}, {"path": "."},
    ])  # fmt: skip
    tools = FakeTools()
    obs = await run_tool_loop(llm, tools, "q", max_calls=3)
    assert len(obs) == 1 and len(tools.calls) == 1
