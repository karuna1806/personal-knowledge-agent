"""Bounded tool use for the `tool` route (§3.7):

1. Tool choice is constrained to an enum of discovered tool names.
2. Arguments are generated under the chosen tool's own JSON schema (constrained decoding),
   then validated with jsonschema before the call.
3. Hard cap of MCP_MAX_TOOL_CALLS calls, then synthesise with whatever was gathered.
"""

import copy
import json
import logging
from dataclasses import dataclass

import jsonschema

from pka.llm.client import LLMClient, Message
from pka.mcp.client import MCPManager, ToolSpec

log = logging.getLogger(__name__)


@dataclass
class ToolObservation:
    tool: str
    arguments: dict
    output: str
    is_error: bool


def _args_schema(spec: ToolSpec) -> dict:
    """The tool's input schema, normalised for constrained decoding."""
    schema = copy.deepcopy(spec.input_schema) or {}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.pop("title", None)
    for prop in schema["properties"].values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    schema["additionalProperties"] = False
    return schema


def _step_schema(tool_names: list[str], allow_finish: bool) -> dict:
    actions = ["call_tool", "answer"] if allow_finish else ["call_tool"]
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": actions},
            "tool": {"type": "string", "enum": tool_names},
        },
        "required": ["action", "tool"],
        "additionalProperties": False,
    }


def _transcript(observations: list[ToolObservation]) -> str:
    if not observations:
        return "No tools have been called yet."
    blocks = []
    for i, obs in enumerate(observations, start=1):
        status = "ERROR" if obs.is_error else "ok"
        blocks.append(
            f"Call {i}: {obs.tool}({json.dumps(obs.arguments)}) → {status}\n{obs.output[:1500]}"
        )
    return "\n\n".join(blocks)


async def run_tool_loop(
    llm: LLMClient,
    tools: MCPManager,
    question: str,
    *,
    max_calls: int,
    memory_hint: str = "",
) -> list[ToolObservation]:
    names = sorted(tools.tools)
    if not names:
        return []
    catalogue = "\n".join(f"- {tools.tools[n].summary()}" for n in names)
    observations: list[ToolObservation] = []
    for step in range(max_calls):
        base: list[Message] = [
            {
                "role": "system",
                "content": "You gather live data with read-only tools to answer the user's question. "
                "Tools:\n" + catalogue + (f"\n\nAbout the user (may be stale): {memory_hint}" if memory_hint else ""),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nSo far:\n{_transcript(observations)}\n\n"
                + (
                    "Pick the single most useful tool to call first."
                    if step == 0
                    else "If the results above already answer the question, choose action=answer. "
                    "Otherwise pick the next tool to call (don't repeat an identical call)."
                ),
            },
        ]
        try:
            decision = await llm.complete_json(base, _step_schema(names, allow_finish=step > 0), name="tool_step", max_tokens=80)
        except Exception as exc:
            log.warning("tool step failed: %s", exc)
            break
        if decision.get("action") == "answer":
            break
        name = decision.get("tool")
        if name not in tools.tools:
            break
        spec = tools.tools[name]
        schema = _args_schema(spec)
        try:
            arguments = await llm.complete_json(
                [
                    *base[:1],
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nSo far:\n{_transcript(observations)}\n\n"
                        f"Write the arguments for {spec.summary()}\n"
                        f"Argument schema: {json.dumps(spec.input_schema.get('properties', {}))}",
                    },
                ],
                schema,
                name="tool_args",
                max_tokens=300,
            )
            jsonschema.validate(arguments, spec.input_schema or {"type": "object"})
        except Exception as exc:
            observations.append(ToolObservation(name, {}, f"could not build valid arguments: {exc}", True))
            continue
        if any(o.tool == name and o.arguments == arguments for o in observations):
            break  # the model is looping on an identical call; stop and answer
        result = await tools.call(name, arguments)
        observations.append(ToolObservation(name, arguments, result.text, result.is_error))
    return observations
