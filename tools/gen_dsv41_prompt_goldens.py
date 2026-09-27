#!/usr/bin/env python3
"""Generates tests/data/dsv41_prompt_goldens.jsonl — the differential goldens
for the DeepSeek-V4.1 prompt renderer (docs/deepseek_v41_flash_plan.md G6).

The GOLDEN SOURCE is the checkpoint's own encoder,
`encoding/encoding_dsv4.py` in the Hub snapshot (no chat_template.jinja
ships with this model), applied to the SERVICE'S template globals:
"messages" (OpenAI messages: system / user / assistant with
reasoning_content and tool_calls / tool with tool_call_id), optional
"tools" (the OpenAI array), "reasoning_effort" (the service's strings),
"clear_thinking" and "enable_thinking". The mapping from the globals to
the reference's call is the renderer's contract, written here once:
  * tools attach to the first system message ("tools" field), or to an
    empty system message inserted at the front;
  * enable_thinking false -> thinking_mode "chat", else "thinking";
  * clear_thinking -> drop_thinking (default true);
  * reasoning_effort: the service's scale folds onto the reference's
    exact levels (minimal/low -> low, medium/high -> high, xhigh/max ->
    max); absent -> the reference's default (low, adds no prefix).
Each case records the render and the tokenizer ids of the render (HF
tokenizers on the snapshot). The header carries the encoder's FNV-1a-64
hash and the tokenizer revision; the C++ gate refuses a different
tokenizer.json.

Regenerating (on spark1, the box with the checkpoint — any venv with
`tokenizers`):
  /home/conrad/audio-bench/venv/bin/python3 tools/gen_dsv41_prompt_goldens.py
      [--model deepseek-ai/DeepSeek-V4-Flash-0731]
      [--snapshot DIR] [--out tests/data/dsv41_prompt_goldens.jsonl]
"""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cluster_doctor import cache_root, cached_snapshot  # noqa: E402
from site_env import cache_environment  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
# The service's coarser scale folds onto the reference's exact levels.
EFFORT = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}


def fnv1a64(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


def weather_tool():
    return {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city (城市天气查询).",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "minimum": 0}}, "required": ["city"]}}}


def search_tool():
    return {"type": "function", "namespace": {"name": "search", "description": "Search tools."}, "function": {
        "name": "lookup", "description": "Look up a value",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}


def call(cid, name, arguments, namespace=None):
    c = {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}
    if namespace is not None:
        c["namespace"] = namespace
    return c


CASES = [
    ("simple_user", {"messages": [{"role": "user", "content": "The capital of France is"}]}),
    ("system_user", {"messages": [{"role": "system", "content": "You are a concise geography assistant."},
                                  {"role": "user", "content": "The capital of France is"}]}),
    ("chat_mode", {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": "What is 2+2?"}], "enable_thinking": False}),
    ("effort_low", {"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "low"}),
    ("effort_high", {"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "high"}),
    ("effort_max", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                    "reasoning_effort": "max"}),
    ("multi_turn_drop_thinking", {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "reasoning_content": "The user said hello, I should greet back.", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "reasoning_content": "It is Paris.", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And of Spain?"}]}),
    ("multi_turn_keep_thinking", {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "reasoning_content": "Greet.", "content": "Hi!"},
        {"role": "user", "content": "Again"}], "clear_thinking": False}),
    ("multi_turn_chat_mode", {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "reasoning_content": "Greet.", "content": "Hi!"},
        {"role": "user", "content": "Again"}], "enable_thinking": False}),
    ("mid_system", {"messages": [
        {"role": "system", "content": "Root."},
        {"role": "user", "content": "One"},
        {"role": "assistant", "content": "Two"},
        {"role": "system", "content": "Now answer in French."}]}),
    ("tools_no_system", {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [weather_tool()]}),
    ("tools_with_system", {"messages": [{"role": "system", "content": "Use tools when helpful."},
                                        {"role": "user", "content": "Weather in Paris for 3 days?"}],
                           "tools": [weather_tool(), search_tool()]}),
    ("tool_call_and_result", {"messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "reasoning_content": "Call the tool.", "content": "",
         "tool_calls": [call("c1", "get_weather", "{\"city\": \"Paris\", \"days\": 3}")]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21, \"unit\": \"C\"}"}],
        "tools": [weather_tool()]}),
    ("parallel_calls_results_reordered", {"messages": [
        {"role": "user", "content": "Weather in Paris and Rome?"},
        {"role": "assistant", "content": "", "tool_calls": [
            call("a", "get_weather", "{\"city\": \"Paris\"}"), call("b", "get_weather", "{\"city\": \"Rome\"}")]},
        {"role": "tool", "tool_call_id": "b", "content": "Rome: 25C"},
        {"role": "tool", "tool_call_id": "a", "content": "Paris: 21C"},
        {"role": "user", "content": "Thanks, summarize."}],
        "tools": [weather_tool()]}),
    ("namespaced_call", {"messages": [
        {"role": "user", "content": "Look it up."},
        {"role": "assistant", "content": "", "tool_calls": [call("s1", "lookup", "{\"query\": \"value\"}", namespace="search")]},
        {"role": "tool", "tool_call_id": "s1", "content": [{"type": "text", "text": "found"}, {"type": "audio", "data": "x"}]}],
        "tools": [search_tool()]}),
    ("call_args_object_and_double_encoded", {"messages": [
        {"role": "user", "content": "Go"},
        {"role": "assistant", "content": "On it.", "tool_calls": [
            call("o1", "get_weather", "{\"city\": \"Oslo\", \"days\": 1, \"flags\": {\"metric\": true}}"),
            call("o2", "get_weather", "\"{\"city\": \"Bergen\"}\""),
            call("o3", "get_weather", "not json")]},
        {"role": "tool", "tool_call_id": "o1", "content": "ok"}],
        "tools": [weather_tool()]}),
    ("content_parts", {"messages": [{"role": "user", "content": "第一段\n\nsecond part",
                                    "content_blocks": [{"type": "text", "text": "第一段"}, {"type": "text", "text": "second part"}]}]}),
    ("assistant_last_no_header", {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "r", "content": "Hello."}]}),
    ("empty_content", {"messages": [{"role": "user", "content": ""}]}),
    ("unicode_and_code", {"messages": [{"role": "system", "content": "Répondez en français, s'il vous plaît."},
                                       {"role": "user", "content": "def add(a, b):\n    return a + b  # 注释\n"}]}),
    # The 0731 reference (encoding_dsv4.py): low effort adds no prefix even
    # with a system message; dict/nested-array arguments render as JSON;
    # with tools present, drop_thinking is disabled so earlier turns keep
    # their reasoning_content.
    ("effort_low_with_system", {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
                                "reasoning_effort": "low"}),
    ("dict_and_nested_array_arguments", {"messages": [
        {"role": "user", "content": "Query the DB."},
        {"role": "assistant", "content": "", "tool_calls": [
            call("d1", "get_weather", "{\"city\": \"Oslo\", \"filters\": {\"tags\": [\"a\", \"b\"], \"limit\": 3, \"on\": true}}")
        ]},
        {"role": "tool", "tool_call_id": "d1", "content": "ok"}],
        "tools": [weather_tool()]}),
    ("reasoning_kept_with_tools", {"messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "reasoning_content": "First I check the weather.", "content": "", "tool_calls": [
            call("w1", "get_weather", "{\"city\": \"Oslo\"}")]},
        {"role": "tool", "tool_call_id": "w1", "content": "Oslo: 12C"},
        {"role": "user", "content": "Thanks. And in Paris?"},
        {"role": "assistant", "reasoning_content": "Now I check Paris.", "content": "Paris is next."}],
        "tools": [weather_tool()]}),
]


def to_reference(kwargs):
    messages = copy.deepcopy(kwargs["messages"])
    tools = kwargs.get("tools")
    if tools:
        if messages and messages[0].get("role") == "system":
            messages[0]["tools"] = tools
        else:
            messages.insert(0, {"role": "system", "content": "", "tools": tools})
    thinking_mode = "chat" if kwargs.get("enable_thinking") is False else "thinking"
    drop = kwargs.get("clear_thinking", True)
    effort = kwargs.get("reasoning_effort")
    if isinstance(effort, str):
        effort = EFFORT[effort]
    return messages, thinking_mode, drop, effort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--out", default="tests/data/dsv41_prompt_goldens.jsonl")
    args = ap.parse_args()
    snap = Path(args.snapshot) if args.snapshot else cached_snapshot(args.model, cache_root(cache_environment()))
    enc_path = snap / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("dsv41_encoding", enc_path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    import tokenizers
    tok_path = snap / "tokenizer.json"
    tok = tokenizers.Tokenizer.from_file(str(tok_path))
    header = {
        "model": args.model,
        "encoder_hash": f"{fnv1a64(enc_path.read_bytes()):016x}",
        "tokenizer_revision": f"{fnv1a64(tok_path.read_bytes()):016x}",
        "tokenizers": tokenizers.__version__,
        "cases": len(CASES),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for name, kwargs in CASES:
            messages, mode, drop, effort = to_reference(kwargs)
            render = enc.encode_messages(messages, thinking_mode=mode, drop_thinking=drop, reasoning_effort=effort)
            ids = tok.encode(render, add_special_tokens=False).ids
            f.write(json.dumps({"name": name, "kwargs": kwargs, "render": render, "ids": ids}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(CASES)} cases (encoder {header['encoder_hash']}, tokenizer {header['tokenizer_revision']})")


if __name__ == "__main__":
    main()
